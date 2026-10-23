# 第三方资源与自有代码授权

本说明适用于选择性同步副本。下文保存主库既有来源与使用记录；本次没有重新核验上游条款或新增分发授权。数据、模型、课程原件和制作期证据不随本批提供。

根目录MIT许可证只适用于本项目自有实现与自有技术文档。它不替换数据、模型、课程原件、第三方代码/文档或服务条款，不把所有目录内容统一改成MIT。用户于2026-10-20选择该自有代码许可。

|资源|来源和用途|适用说明|
|---|---|---|
|CMeEE-V2 / CBLUE|[官方数据页](https://tianchi.aliyun.com/dataset/95414)，训练和参考标注|页面为CC BY-NC-SA 4.0；非商业教学，署名及改编边界见[数据许可记录](docs/DATA_LICENSE.md)。不随源码包再分发数据或规范附件|
|中文BERT|[google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)，N1编码器初始化|官方模型卡标Apache-2.0，固定实际下载版本记录在evidence/stage2/model-download.json；导出权重保留来源，不能因自有代码MIT而变更模型/数据义务|
|BERT方法|[Devlin et al.,2019](https://aclanthology.org/N19-1423/)|预训练方法不是本项目原创|
|GlobalPointer方法|[Su et al.,2022](https://arxiv.org/abs/2208.03054)|跨度矩阵和头部RoPE依据该方法独立实现；不把方法思想归为原创，不含未经核查复制的作者仓库代码|
|PyTorch|torch 2.8.0|其发行包自带BSD-style许可和第三方NOTICE，保留上游声明|
|Transformers / safetensors / TensorBoard|锁文件固定发行包|各自Apache-2.0许可与附属声明，采用包内许可，不在本项目MIT下再授权|
|NumPy / jieba|锁文件固定发行包|NumPy BSD-3-Clause、jieba MIT；保留各发行包声明|
|DeepSeek官方API|固定零/三样本抽取，调用时模型/价格见原始记录|商业服务条款单独适用；模型响应不是本项目原创人工标注，不能假定任意再分发权限或供应商零留存|
|课程任务书/模板/贡献表|仅private/course-originals用于本课核对|不随公开源码重新分发，不自行改写为官方新规则|

数据引用：Zhang et al.(2022), CBLUE: A Chinese Biomedical Language Understanding Evaluation Benchmark, ACL 7888–7915，[论文](https://aclanthology.org/2022.acl-long.544/)。语料原始来源的其他引用按官方README保留。

最终源码ZIP排除data、models、private、虚拟环境、缓存、视频及含完整原文的批量预测。模型和数据通过各自许可允许的独立方式取得。本文件是本项目使用记录，不是新增第三方授权证明。
