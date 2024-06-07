Network: ConditionalUnet1D
    args: 
        input_dim 7
        global_cond_dim (512+7)x2
        diffusion_step_embed_dim 256
        down_dims [256,512,1024]
        kernel_size 5
        n_groups 8

    init:
        args:
            all_dims=[7,256,512,1024]
            start_dim=256
            dsed=diffusion_step_embed_dim
            cond_dim=dsed+global_cond_dim=256+1038=1294
            in_out=[(7,256),(256,512),(512,1024)]
            mid_dim=1024
        model:
            diffusion_step_encoder:
                SinusoidalPosEmb(dsed)
                nn.Linear(dsed, dsed * 4)
                nn.Mish()
                nn.Linear(dsed * 4, dsed)
            mid_modules:
                ConditionalResidualBlock1D(
                    mid_dim,mid_dim,cond_dim,kernel_size,n_groups)
                ConditionalResidualBlock1D(
                    mid_dim,mid_dim,cond_dim,kernel_size,n_groups)
            down_modules:
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups)
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups), 
                Downsample1d(dim_out) if not is_last
            up_modules:
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups)
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last
            final_conv:
                Conv1dBlock(
                    start_dim, start_dim, kernel_size=kernel_size)
                nn.Conv1d(
                    start_dim, input_dim, 1)


