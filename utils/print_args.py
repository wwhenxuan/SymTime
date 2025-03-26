def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}')
    print(f'  {"Dataset:":<20}{args.dataset_name:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(
        f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}'
    )
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.task_name == "long_term_forecast":
        print("\033[1m" + "Long-term Forecasting Task" + "\033[0m")
        print(
            f'  {"Seq Len:":<20}{args.seq_len:<20}{"Pred Len:":<20}{args.pred_len:<20}'
        )
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    if args.task_name == "short_term_forecast":
        print("\033[1m" + "short-term Forecasting Task" + "\033[0m")
        print(f'  {"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    if args.task_name == "imputation":
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        print()

    if args.task_name == "anomaly_detection":
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(
        f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}'
    )
    print(
        f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}'
    )
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(
        f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}'
    )
    print()
