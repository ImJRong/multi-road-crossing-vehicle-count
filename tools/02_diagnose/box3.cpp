// box3: 三个模型跑同一帧, 和 FP32 真值比, 输出每个检出框的误差.
//
// 板子上的模型对比真值用. 真值是 PC 上 onnxruntime 跑 FP32 ONNX 的结果.
//
// 用法:
//   /tmp/box3 v7.rknn head4.rknn video.mp4 frame_no truth.txt
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"

namespace {

// 和板子 main.cpp 保持一致：ROI 外接矩形裁切 + letterbox
const int IMGSZ = 640;
const int ROI_X1 = 278, ROI_Y1 = 90, ROI_X2 = 984, ROI_Y2 = 541;
const float CONF = 0.4f;

struct Box {
  float cx, cy, w, h, score;
  float x1() const { return cx - w / 2; }
  float y1() const { return cy - h / 2; }
  float x2() const { return cx + w / 2; }
  float y2() const { return cy + h / 2; }
};

float iou(const Box& a, const Box& b) {
  const float ix = std::min(a.x2(), b.x2()) - std::max(a.x1(), b.x1());
  const float iy = std::min(a.y2(), b.y2()) - std::max(a.y1(), b.y1());
  if (ix <= 0 || iy <= 0) return 0.f;
  const float inter = ix * iy;
  const float ua = a.w * a.h + b.w * b.h - inter;
  return ua > 0 ? inter / ua : 0.f;
}

struct Model {
  rknn_context ctx = 0;
  std::vector<unsigned char> buf;
  bool is_dist = false;
  int boxes_idx = 0, scores_idx = 1, n_cols = 8400;
  std::vector<float> ax, ay, st;
  std::string tag;

  bool load(const char* path, const char* name) {
    tag = name;
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
      for (const auto& s : segs)
        for (int j = 0; j < s.n && k < n_cols; ++j, ++k) {
          ax[static_cast<size_t>(k)] = static_cast<float>(j % s.g) + 0.5f;
          ay[static_cast<size_t>(k)] = static_cast<float>(j / s.g) + 0.5f;
          st[static_cast<size_t>(k)] = s.s;
        }
    }
    return true;
  }

  std::vector<Box> run(const cv::Mat& rgb, double s, int dx, int dy) {
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
      if (score[i] < CONF) continue;
      float x1, y1, x2, y2;
      if (is_dist) {
        const size_t k = static_cast<size_t>(i);
        const float sr = st[k];
        x1 = (ax[k] - coord[k]) * sr;
        y1 = (ay[k] - coord[(size_t)n_cols + k]) * sr;
        x2 = (ax[k] + coord[(size_t)2 * n_cols + k]) * sr;
        y2 = (ay[k] + coord[(size_t)3 * n_cols + k]) * sr;
      } else {
        const float cx = coord[i], cy = coord[(size_t)n_cols + i];
        const float w = coord[(size_t)2 * n_cols + i], h = coord[(size_t)3 * n_cols + i];
        x1 = cx - w / 2; y1 = cy - h / 2; x2 = cx + w / 2; y2 = cy + h / 2;
      }
      // 640 空间 -> 原图，加回 ROI 原点（和 main.cpp 的 d.x1 += roi_rect.x 一致）
      Box b;
      b.cx = ((x1 + x2) / 2 - dx) / s + ROI_X1;
      b.cy = ((y1 + y2) / 2 - dy) / s + ROI_Y1;
      b.w = (x2 - x1) / s;
      b.h = (y2 - y1) / s;
      b.score = score[i];
      out.push_back(b);
    }
    rknn_outputs_release(ctx, io.n_output, outs.data());
    return out;
  }

  ~Model() { if (ctx) rknn_destroy(ctx); }
};

std::vector<Box> read_truth(const char* path) {
  std::vector<Box> v;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    float x1, y1, x2, y2, cx, cy, w, h, sc;
    if (!(ss >> x1 >> y1 >> x2 >> y2 >> cx >> cy >> w >> h >> sc)) continue;
    Box b; b.cx = cx; b.cy = cy; b.w = w; b.h = h; b.score = sc;
    v.push_back(b);
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::printf("用法: %s a.rknn b.rknn video.mp4 frame_no truth.txt\n", argv[0]);
    return 1;
  }
  const int frame_no = std::atoi(argv[4]);
  auto truth = read_truth(argv[5]);
  std::printf("真值 %zu 个框\n", truth.size());

  cv::VideoCapture cap(argv[3]);
  if (!cap.isOpened()) { std::printf("打不开视频\n"); return 1; }
  cap.set(cv::CAP_PROP_POS_FRAMES, frame_no);
  cv::Mat src;
  if (!cap.read(src) || src.empty()) { std::printf("读帧失败\n"); return 1; }
  cap.release();

  // 和 main.cpp 一样：先裁 ROI，再对裁出来的图 letterbox
  cv::Mat crop = src(cv::Rect(ROI_X1, ROI_Y1, ROI_X2 - ROI_X1, ROI_Y2 - ROI_Y1));
  const double s = std::min(IMGSZ / static_cast<double>(crop.cols),
                            IMGSZ / static_cast<double>(crop.rows));
  const int nw = static_cast<int>(std::round(crop.cols * s));
  const int nh = static_cast<int>(std::round(crop.rows * s));
  const int dx = (IMGSZ - nw) / 2, dy = (IMGSZ - nh) / 2;
  cv::Mat rs, canvas(IMGSZ, IMGSZ, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(crop, rs, cv::Size(nw, nh));
  rs.copyTo(canvas(cv::Rect(dx, dy, nw, nh)));
  cv::Mat rgb;
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
  std::printf("ROI %dx%d -> %dx%d  scale=%.4f pad=(%d,%d)\n", crop.cols, crop.rows, nw, nh, s, dx, dy);

  Model ma, mb;
  if (!ma.load(argv[1], "A") || !mb.load(argv[2], "B")) return 1;

  auto ba = ma.run(rgb, s, dx, dy);
  auto bb = mb.run(rgb, s, dx, dy);

  auto report = [&](const std::vector<Box>& det, const char* name) {
    std::printf("\n===== %s  检出 %zu 个 =====\n", name, det.size());
    std::printf("%-4s %-22s %-22s %-24s\n", "配对", "真值(cx,cy,w,h)", "检出(cx,cy,w,h)", "误差 左/上/宽/高");
    int n = 0;
    double sl = 0, st_ = 0, sw = 0, sh = 0, scx = 0, scy = 0;
    for (const auto& t : truth) {
      const Box* best = nullptr;
      float bi = 0.25f;
      for (const auto& d : det) {
        const float v = iou(t, d);
        if (v > bi) { bi = v; best = &d; }
      }
      if (!best) continue;
      const double dl = best->x1() - t.x1();
      const double dt = best->y1() - t.y1();
      const double dw = best->w - t.w;
      const double dh = best->h - t.h;
      std::printf("%-4d (%6.1f,%6.1f,%5.1f,%5.1f) (%6.1f,%6.1f,%5.1f,%5.1f) (%+6.1f,%+6.1f,%+6.1f,%+6.1f)\n",
                  n, t.cx, t.cy, t.w, t.h, best->cx, best->cy, best->w, best->h, dl, dt, dw, dh);
      sl += dl; st_ += dt; sw += dw; sh += dh; scx += best->cx - t.cx; scy += best->cy - t.cy;
      ++n;
    }
    if (n > 0) {
      std::printf("---- 平均误差: 左 %+.2f  上 %+.2f  宽 %+.2f  高 %+.2f  中心偏(%+.2f,%+.2f)\n",
                  sl / n, st_ / n, sw / n, sh / n, scx / n, scy / n);
    } else {
      std::printf("---- 没配上对\n");
    }
    return n;
  };

  const int na = report(ba, argv[1]);
  const int nb = report(bb, argv[2]);
  std::printf("\n(配对数: A=%d  B=%d)\n", na, nb);
  return 0;
}
