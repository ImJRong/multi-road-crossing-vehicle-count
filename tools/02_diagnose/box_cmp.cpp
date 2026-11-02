// box_cmp: 同一个视频帧, 两个模型各跑一次, 把检测框按 IoU 配对后逐框比较.
//
// 目的: 量化 head4(距离换算) 和 v7(旧) 的框差多少像素.
// 用法:
//   /tmp/box_cmp a.rknn b.rknn video.mp4 frame_no
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"

namespace {

const int IMGSZ = 640;

struct Box {
  float x1, y1, x2, y2, score;
};

float iou(const Box& a, const Box& b) {
  const float ix = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const float iy = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (ix <= 0 || iy <= 0) return 0.f;
  const float inter = ix * iy;
  const float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0.f;
}

struct Model {
  rknn_context ctx = 0;
  std::vector<unsigned char> buf;
  bool is_dist = false;
  int boxes_idx = 0, scores_idx = 1, n_cols = 8400;
  std::vector<float> ax, ay, st;

  bool load(const char* path) {
    FILE* fp = std::fopen(path, "rb");
    if (!fp) { std::printf("打不开 %s\n", path); return false; }
    std::fseek(fp, 0, SEEK_END);
    const long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    buf.resize(static_cast<size_t>(sz));
    if (std::fread(buf.data(), 1, buf.size(), fp) != buf.size()) { std::fclose(fp); return false; }
    std::fclose(fp);
    if (rknn_init(&ctx, buf.data(), static_cast<uint32_t>(buf.size()), 0, nullptr) != 0) {
      std::printf("rknn_init 失败 %s\n", path);
      return false;
    }
    rknn_input_output_num io;
    std::memset(&io, 0, sizeof(io));
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    int cols = 0;
    for (uint32_t i = 0; i < io.n_output; ++i) {
      rknn_tensor_attr a;
      std::memset(&a, 0, sizeof(a));
      a.index = i;
      rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
      if (a.n_dims < 3) continue;
      const int r = a.dims[a.n_dims - 2];
      const int c = a.dims[a.n_dims - 1];
      if (c > cols) cols = c;
      if (r == 4) {
        boxes_idx = static_cast<int>(i);
        if (std::string(a.name).find("dfl") != std::string::npos) is_dist = true;
      } else if (r > 0) {
        scores_idx = static_cast<int>(i);
      }
    }
    n_cols = cols;
    if (is_dist) {
      struct S { int n, g; float s; };
      const S segs[3] = {{6400, 80, 8.f}, {1600, 40, 16.f}, {400, 20, 32.f}};
      ax.resize(static_cast<size_t>(n_cols));
      ay.resize(static_cast<size_t>(n_cols));
      st.resize(static_cast<size_t>(n_cols));
      int k = 0;
      for (const auto& s : segs) {
        for (int j = 0; j < s.n && k < n_cols; ++j, ++k) {
          ax[static_cast<size_t>(k)] = static_cast<float>(j % s.g) + 0.5f;
          ay[static_cast<size_t>(k)] = static_cast<float>(j / s.g) + 0.5f;
          st[static_cast<size_t>(k)] = s.s;
        }
      }
    }
    return true;
  }

  std::vector<Box> run(const cv::Mat& rgb, float conf_th) {
    rknn_input in;
    std::memset(&in, 0, sizeof(in));
    in.index = 0;
    in.buf = const_cast<unsigned char*>(rgb.data);
    in.size = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
    in.pass_through = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    rknn_inputs_set(ctx, 1, &in);
    rknn_run(ctx, nullptr);

    rknn_input_output_num io;
    std::memset(&io, 0, sizeof(io));
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    std::vector<rknn_output> outs(io.n_output);
    for (uint32_t i = 0; i < io.n_output; ++i) {
      std::memset(&outs[i], 0, sizeof(rknn_output));
      outs[i].index = i;
      outs[i].want_float = 1;
      outs[i].is_prealloc = 0;
    }
    rknn_outputs_get(ctx, io.n_output, outs.data(), nullptr);

    const float* coord = static_cast<const float*>(outs[static_cast<size_t>(boxes_idx)].buf);
    const float* score = static_cast<const float*>(outs[static_cast<size_t>(scores_idx)].buf);

    std::vector<Box> out;
    for (int i = 0; i < n_cols; ++i) {
      if (score[i] < conf_th) continue;
      float x1, y1, x2, y2;
      if (is_dist) {
        const size_t k = static_cast<size_t>(i);
        const float s = st[k];
        x1 = (ax[k] - coord[k]) * s;
        y1 = (ay[k] - coord[(size_t)n_cols + k]) * s;
        x2 = (ax[k] + coord[(size_t)2 * n_cols + k]) * s;
        y2 = (ay[k] + coord[(size_t)3 * n_cols + k]) * s;
      } else {
        const float cx = coord[i], cy = coord[(size_t)n_cols + i];
        const float w = coord[(size_t)2 * n_cols + i], h = coord[(size_t)3 * n_cols + i];
        x1 = cx - w / 2; y1 = cy - h / 2;
        x2 = cx + w / 2; y2 = cy + h / 2;
      }
      out.push_back({x1, y1, x2, y2, score[i]});
    }
    rknn_outputs_release(ctx, io.n_output, outs.data());
    return out;
  }

  ~Model() { if (ctx) rknn_destroy(ctx); }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("用法: %s a.rknn b.rknn video.mp4 [frame_no]\n", argv[0]);
    return 1;
  }
  const int frame_no = argc > 4 ? std::atoi(argv[4]) : 200;

  cv::VideoCapture cap(argv[3]);
  if (!cap.isOpened()) { std::printf("打不开视频\n"); return 1; }
  cap.set(cv::CAP_PROP_POS_FRAMES, frame_no);
  cv::Mat src;
  if (!cap.read(src) || src.empty()) { std::printf("读帧失败\n"); return 1; }
  cap.release();

  const double s = std::min(IMGSZ / static_cast<double>(src.cols), IMGSZ / static_cast<double>(src.rows));
  const int nw = static_cast<int>(std::round(src.cols * s));
  const int nh = static_cast<int>(std::round(src.rows * s));
  cv::Mat rs, canvas(IMGSZ, IMGSZ, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(src, rs, cv::Size(nw, nh));
  rs.copyTo(canvas(cv::Rect((IMGSZ - nw) / 2, (IMGSZ - nh) / 2, nw, nh)));
  cv::Mat rgb;
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

  Model ma, mb;
  if (!ma.load(argv[1]) || !mb.load(argv[2])) return 1;

  const float TH = 0.4f;
  auto ba = ma.run(rgb, TH);
  auto bb = mb.run(rgb, TH);
  std::printf("\nA=%s  检出 %zu 个 (>%.1f)\n", argv[1], ba.size(), TH);
  std::printf("B=%s  检出 %zu 个 (>%.1f)\n", argv[2], bb.size(), TH);

  // 按 IoU 配对
  std::vector<int> used(bb.size(), 0);
  std::printf("\n%-4s %-28s %-28s %s\n", "配对", "A 框 (cx,cy,w,h)", "B 框 (cx,cy,w,h)", "差(左,上,右,下)");
  std::printf("%s\n", std::string(100, '-').c_str());
  int paired = 0;
  double sum_dl = 0, sum_dt = 0, sum_dw = 0, sum_dh = 0;
  for (const auto& a : ba) {
    int best_j = -1;
    float best_iou = 0.3f;
    for (size_t j = 0; j < bb.size(); ++j) {
      if (used[j]) continue;
      const float v = iou(a, bb[j]);
      if (v > best_iou) { best_iou = v; best_j = static_cast<int>(j); }
    }
    if (best_j < 0) continue;
    used[static_cast<size_t>(best_j)] = 1;
    const Box& b = bb[static_cast<size_t>(best_j)];
    const float acx = (a.x1 + a.x2) / 2, acy = (a.y1 + a.y2) / 2;
    const float aw = a.x2 - a.x1, ah = a.y2 - a.y1;
    const float bcx = (b.x1 + b.x2) / 2, bcy = (b.y1 + b.y2) / 2;
    const float bw = b.x2 - b.x1, bh = b.y2 - b.y1;
    const float dl = b.x1 - a.x1, dt = b.y1 - a.y1, dr = b.x2 - a.x2, db = b.y2 - a.y2;
    std::printf("%-4d (%6.1f,%6.1f,%5.1f,%5.1f) (%6.1f,%6.1f,%5.1f,%5.1f) "
                "(%+5.1f,%+5.1f,%+5.1f,%+5.1f)\n",
                paired, acx, acy, aw, ah, bcx, bcy, bw, bh, dl, dt, dr, db);
    sum_dl += dl; sum_dt += dt; sum_dw += (bw - aw); sum_dh += (bh - ah);
    ++paired;
  }

  std::printf("%s\n", std::string(100, '-').c_str());
  if (paired > 0) {
    std::printf("配对 %d 个, 平均差: 左边缘 %+.2f  上边缘 %+.2f  宽 %+.2f  高 %+.2f\n",
                paired, sum_dl / paired, sum_dt / paired, sum_dw / paired, sum_dh / paired);
  } else {
    std::printf("没有配上对（两模型检测差异太大）\n");
  }
  return 0;
}
