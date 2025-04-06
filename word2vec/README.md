## Vocabulary类
1. 在Vocabulary类中，mask_token对应的索引通过调用add_token方法赋值给self.<u>mask_index</u>属性。
2. lookup_token方法中，如果self.unk_index >=0，则对未登录词返回<u>self.unk_index</u>。
3. 调用add_many方法添加多个token时，实际是通过循环调用<u>add_token</u>方法实现的。

## CBOWVectorizer类
4. vectorize方法中，当vector_length < 0时，最终向量长度等于<u>indices</u>的长度。
5. from_dataframe方法构建词表时，会遍历DataFrame中<u>context</u>和<u>target</u>两列的内容。
6. out_vector[len(indices):]的部分填充为self.cbow_vocab.<u>mask_index</u>。

## CBOWDataset类
7. _max_seq_length通过计算所有context列的<u>分词后的单词数量</u>的最大值得出。
8. set_split方法通过self._lookup_dict选择对应的<u>DataFrame</u>和<u>size</u>。
9. __getitem__返回的字典中，y_target通过查找<u>target</u>列的token得到。

## 模型结构
10. CBOWClassifier的forward中，x_embedded_sum的计算方式是embedding(x_in).<u>sum</u>(dim=1)。
11. 模型输出层fc1的out_features等于<u>vocabulary_size</u>参数的值。

## 训练流程
12. generate_batches函数通过PyTorch的<u>DataLoader</u>类实现批量加载。
13. 训练时classifier.train()的作用是启用<u>dropout</u>和<u>batch normalization</u>模式。
14. 反向传播前必须执行<u>optimizer</u>.zero_grad()清空梯度。
15. compute_accuracy中y_pred_indices通过<u>max(dim=1)</u>方法获取预测类别。

## 训练状态管理
16. make_train_state中early_stopping_best_val初始化为<u>1e8</u>。
17. update_train_state在连续<u>5</u>次验证损失未下降时会触发早停。
18. 当验证损失下降时，early_stopping_step会被重置为<u>0</u>。

## 设备与随机种子
19. set_seed_everywhere中与CUDA相关的设置是<u>torch.cuda</u>.manual_seed_all(seed)。
20. args.device的值根据<u>torch.cuda</u>.is_available()确定。

## 推理与测试
21. get_closest函数中排除计算的目标词本身是通过continue判断word == <u>target_word</u>实现的。
22. 测试集评估时一定要调用<u>classifier.eval()</u>方法禁用dropout。

## 关键参数
23. CBOWClassifier的padding_idx参数默认值为<u>0</u>。
24. args中控制词向量维度的参数是<u>embedding_size</u>。
25. 学习率调整策略ReduceLROnPlateau的触发条件是验证损失<u>增加</u>（增加/减少）。