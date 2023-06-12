import argparse
import os
from recbole.quick_start import objective_function
from recbole.trainer import HyperTuning


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_files", type=str, default=None, help="fixed config files")
    parser.add_argument("--params_file", type=str, default=None, help="parameters file")
    # parser.add_argument("--cv", type=int, default=5, help="number of cross-validation folds")
    args, _ = parser.parse_known_args()

    # 하이퍼파라미터 그리드, config파일 입력 
    
    args.config_files = os.path.join("./config", args.config_files)
    args.params_file = os.path.join("./hyper", args.params_file)

    # Run hyperparameter tuning with cross-validation
    hp = HyperTuning(
        objective_function,
        algo="exhaustive",  # random, exhaustive
        params_file=args.params_file,
        fixed_config_file_list=[args.config_files]
        # cv=args.cv
    )
    hp.run()

    # Print the best hyperparameter configuration
    best_params = hp.best_params
    best_result = hp.params2result[hp.params2str(hp.best_params)]

    print("\n===== Best Result =====")
    print(f"Best params: {best_params}")
    print(f"Best result: {best_result}") 