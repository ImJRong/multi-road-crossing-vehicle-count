// ch_scan: 打印 boxes 输出 4 个通道各自的范围, 判断模型是否坏了.
//
// 好模型 4 个通道范围各不相同 (cx/cy 覆盖全图, w/h 只到几百):
//   ch0(cx) 0~630 | ch1(cy) 0~615 | ch2(w) 2.5~510 | ch3(h) 2.5~230
// 坏模型 4 个通道范围一样 (都是 0~638) = 通道语义被破坏.
//
// 编译(板子上):
//   g++ -O2 -std=c++17 -I~/code/src /tmp/ch_scan.cpp -o /tmp/ch_scan \
//       -lrknnrt $(pkg-config --cflags --libs opencv4)
// 用法:
//   /tmp/ch_scan <model.rknn> <video.mp4> [frame_no] [conf]
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

const int N_ANCHOR = 8400;
const int IMGSZ = 640;

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("用法: %s <model.rknn> [video.mp4] [frame_no] [conf]\n", argv[0]);
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string video = argc > 2 ? argv[2] : "";
  const int frame_no = argc > 3 ? std::atoi(argv[3]) : 200;
  const float conf_th = argc > 4 ? static_cast<float>(std::atof(argv[4])) : 0.25f;

  FILE* fp = std::fopen(model_path.c_str(), "rb");
  if (!fp) { std::printf("打不开模型\n"); return 1; }
  std::fseek(fp, 0, SEEK_END);
  const long sz = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  std::vector<unsigned char> mbuf(static_cast<size_t>(sz));
  if (std::fread(mbuf.data(), 1, mbuf.size(), fp) != mbuf.size()) { std::fclose(fp); return 1; }
  std::fclose(fp);

  rknn_context ctx = 0;
  if (rknn_init(&ctx, mbuf.data(), static_cast<uint32_t>(mbuf.size()), 0, nullptr) != 0) {
    std::printf("rknn_init 失败\n");
    return 1;
  }

  rknn_input_output_num io;
  std::memset(&io, 0, sizeof(io));
  if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != 0) return 1;

  std::vector<rknn_tensor_attr> out(io.n_output);
  for (uint32_t i = 0; i < io.n_output; ++i) {
    std::memset(&out[i], 0, sizeof(rknn_tensor_attr));
    out[i].index = i;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out[i], sizeof(rknn_tensor_attr));
  }

  // ---- 造输入: 整帧 letterbox 到 640 ----
  cv::Mat frame(IMGSZ, IMGSZ, CV_8UC3, cv::Scalar(114, 114, 114));
  double s = 1.0;
  int dx = 0, dy = 0;
  if (!video.empty()) {
    cv::VideoCapture cap(video);
    if (cap.isOpened()) {
      cap.set(cv::CAP_PROP_POS_FRAMES, frame_no);
      cv::Mat src;
      if (cap.read(src) && !src.empty()) {
        s = std::min(IMGSZ / static_cast<double>(src.cols), IMGSZ / static_cast<double>(src.rows));
        const int nw = static_cast<int>(std::round(src.cols * s));
        const int nh = static_cast<int>(std::round(src.rows * s));
        cv::Mat rs;
        cv::resize(src, rs, cv::Size(nw, nh));
        dx = (IMGSZ - nw) / 2;
        dy = (IMGSZ - nh) / 2;
        rs.copyTo(frame(cv::Rect(dx, dy, nw, nh)));
        std::printf("输入: %s 第 %d 帧 %dx%d -> %dx%d  pad(%d,%d)\n", video.c_str(), frame_no,
                    src.cols, src.rows, nw, nh, dx, dy);
      }
    }
  }
  cv::Mat rgb;
  cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

  rknn_input inputs[1];
  std::memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].buf = rgb.data;
  inputs[0].size = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
  inputs[0].pass_through = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  if (rknn_inputs_set(ctx, 1, inputs) != 0) { std::printf("inputs_set 失败\n"); return 1; }
  if (rknn_run(ctx, nullptr) != 0) { std::printf("run 失败\n"); return 1; }

  std::vector<rknn_output> outs(io.n_output);
  for (uint32_t i = 0; i < io.n_output; ++i) {
    std::memset(&outs[i], 0, sizeof(rknn_output));
    outs[i].index = i;
    outs[i].want_float = 1;
    outs[i].is_prealloc = 0;
  }
  if (rknn_outputs_get(ctx, io.n_output, outs.data(), nullptr) != 0) {
    std::printf("outputs_get 失败\n");
    return 1;
  }

  const float* boxes = static_cast<const float*>(outs[0].buf);
  const float* scores = static_cast<const float*>(outs[1].buf);

  // ---- 每个通道独立统计 ----
  static const char* kName[4] = {"cx", "cy", "w ", "h "};
  std::printf("\n===== boxes 4 个通道各自范围 =====\n");
  for (int c = 0; c < 4; ++c) {
    float mn = boxes[c * N_ANCHOR], mx = boxes[c * N_ANCHOR];
    double sum = 0;
    for (int i = 0; i < N_ANCHOR; ++i) {
      const float v = boxes[c * N_ANCHOR + i];
      mn = std::min(mn, v);
      mx = std::max(mx, v);
      sum += v;
    }
    std::printf("  ch%d(%s) %8.2f ~ %8.2f   均值 %7.2f\n", c, kName[c], mn, mx, sum / N_ANCHOR);
  }

  // ---- 只看分数高的那些 anchor, 它们的坐标才有意义 ----
  std::vector<int> keep;
  for (int i = 0; i < N_ANCHOR; ++i) {
    if (scores[i] >= conf_th) keep.push_back(i);
  }
  std::printf("\n===== 分数 >= %.2f 的 %zu 个 anchor 的坐标 =====\n", conf_th, keep.size());
  for (size_t k = 0; k < keep.size() && k < 20; ++k) {
    const int i = keep[k];
    const float cx = boxes[0 * N_ANCHOR + i];
    const float cy = boxes[1 * N_ANCHOR + i];
    const float w = boxes[2 * N_ANCHOR + i];
    const float h = boxes[3 * N_ANCHOR + i];
    // 反 letterbox 回原图
    const float ox = (cx - dx) / s;
    const float oy = (cy - dy) / s;
    const float ow = w / s;
    const float oh = h / s;
    std::printf("  #%-5d conf=%.3f  640空间 cx=%6.1f cy=%6.1f w=%6.1f h=%6.1f"
                "  原图 cx=%6.1f cy=%6.1f w=%6.1f h=%6.1f\n",
                i, scores[i], cx, cy, w, h, ox, oy, ow, oh);
  }

  // ---- 和"上一帧/下一帧"无关的纯几何检查: 框中心是否落在图内 ----
  int inside = 0, total = 0;
  for (int i : keep) {
    const float cx = boxes[0 * N_ANCHOR + i];
    const float cy = boxes[1 * N_ANCHOR + i];
    if (cx >= 0 && cx <= IMGSZ && cy >= 0 && cy <= IMGSZ) ++inside;
    ++total;
  }
  std::printf("\n  中心在 640x640 图内的: %d / %d\n", inside, total);

  rknn_outputs_release(ctx, io.n_output, outs.data());
  rknn_destroy(ctx);
  return 0;
}
