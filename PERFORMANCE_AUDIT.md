# 2026-10-16 流水线性能核查

本文件记录早期阶段；当前可直接运行的打包入口和验证状态见 [GPU_RUNNER.md](docs/GPU_RUNNER.md)、[ACCELERATION.md](docs/ACCELERATION.md)、[VALIDATION.md](docs/VALIDATION.md)。下文提到的工作区报告/数组是历史外部材料，不随仓库分发。完整 agent 历史记录见 [HISTORICAL_HANDOFF.md](docs/HISTORICAL_HANDOFF.md)。

全部实测在用户 4 号机完成，CPU 配额 12 核、内存配额 72 GiB，使用该机 CUDA GPU。几何和镜像计算保留 FP64，距离场输出 FP32；没有插值、平滑或降低输出精度。

## 已接入本地仓库的部分

`pipeline/stage3_middata/wwv_numeric.py` 实现 WWV 数值编译版本。`build_wwv.py` 通过 `Config.WWV_BACKEND` 选择后端：

- `"numba"`：本地默认，Numba 编译并用线程执行，内层线程数采用 `INNER_WORKERS`；关闭 `USE_INNER_MP` 时使用一个线程，`INNER_WORKERS=None` 时也保守使用一个线程。
- `"python"`：保留原 Python / 多进程参考实现。

事件仍按 `(angle, wall_id)` 排序，角区间、EPS、端点偏移方向及对称化规则保持不变。活动墙改为数值数组，同一射线方向只计算一次三角函数。距离完全并列的查询回到原 Python 实现，保留历史 set 遍历造成的并列选择行为。首次 JIT 另需时间；后续调用及磁盘编译缓存可复用。

4 号机验证了 5 个真实城市的整个 Stage 3 WWV 加载、计算和保存接口，全部矩阵与原实现一致；另有 12 个重复墙、相交墙、零长度墙等退化几何用例和空几何验证。测试材料见工作区 `precision_audit/pipeline_optimization/reports/`。此改动只写本地仓库，未推送远端。

## 已验证、尚在实验入口的部分

工作区 `precision_audit/pipeline_*.py` 中保留 GPU PVP / 多边形栅格化、常驻城市数据求解器、显式传入源点缓存、编译源点预处理及位压缩到 CSR 的实验。GPU 方案尚未接入仓库的 `run_all.py`，也没有启动 701 城市生产任务。

场景 1 的 GPU/CPU 混合实验重新构建了 2560×2560 的 PVW、PVP 和建筑掩码，然后计算 16 个发射点距离场。可见性缓存与原缓存逐字节一致，16 张距离场与此前高分辨率结果逐点一致。实验仍使用已有 WWV；没有改变路径规则。完整范围、耗时、内存和局限见工作区 `precision_audit/pipeline_optimization/README.md`。

原始旧入口中的 PNG 几何预处理、object 中间格式转换和逐源子进程调度，与本次官方多边形实验入口不是同一个生产流程。不能仅启用 WWV 后端，就声称旧 `run_all.py` 已获得整个 GPU 实验的提速。

## 第二轮更新：完整 GPU 求解与精确角度排序

4、6号机分别完成了场景1的16源完整2560×2560 GPU距离场验证。4号机另完成5城市23张256×256场对照。GPU直接读压缩PVW/PVP，保留FP64几何/长度和FP32输出；当前生成内核只验证K=1。完整GPU流程仍是实验入口，尚未替换 `run_all.py`。

本地仓库新增 `utils/angular_events.py`，由 `build_pvw.py` 调用。先稳定按角度排序，再在相邻角度差小于1e-12的连通组内重放原输入顺序的插入比较器，保持原容差与非传递比较行为。3000组排序回归和5城市原函数对照通过；不改变墙可见性判据。

4号机同一城市16源的完整流程，上一轮163.873秒；完整GPU路径、原墙排序76.863秒；再采用精确分组排序48.812秒。所有可见性缓存和16张距离场与参考逐点一致。最终峰值进程RSS约3082MiB。6号机独立验证及未采用的GPU排序实验单独保留，没有用跨机器差异计算提速。

代码、计时范围、回归报告与复现命令见工作区 `precision_audit/round2_results/README.md`。这是单城高分辨率完整流程实验，不表示已完成701城批量生产适配或验证。
