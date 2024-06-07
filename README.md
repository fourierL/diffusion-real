注意事项：如果做多任务，每个任务收集的轨迹数量必须一致。
需要自行新建一个clip_model文件夹，从huggingface上把相关配置文件和参数下载到该文件中

step0: 准备多任务的语言指令，config文件夹下的instr.json中定义，保证每个任务对应的语言指令数量相同。执行preprocess下的load_instructions,使用clip模型将语言转为特征。


step1: 将收集的数据data_drawer_npy放到rlbench_data文件夹中，执行preprocess文件夹下的process_pose_and_image.py，将每条轨迹在step维度上进行叠加，比如images/episode0下面有30个step,将其处理为30xNxNxC维度。
再执行data_process.py，将所有轨迹在step维度上进行叠加，如果有30条轨迹，每条轨迹50步，则会得到30x50xNxNxC的图像数据，state数据同理。同时执行dat_process.py可以对所有step进行任务分类。

step2: train_rec.py开始训练,1

step3: eval_rec.py尚待加入语言多任务