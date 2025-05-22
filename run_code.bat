@echo off
setlocal

REM Define lists for each parameter
set init_seed_list=1
set dataset_list=cifar10
set batchsize_list=128
set cd_lr_list=0.001
set partition_list=XtremeHetero
set epochs_classifier_list=100
set n_parties_list=10
set algorithm_list=fedeov fedov
set rounds_list=100
set beta_list=0.1
set pruneratio_list=0
set model_list=SimpleCNN2
set device=cuda:0

REM Define flags as True or False
set save_result_dict=True
set save_classifier=True
set ablation=True

REM Add flags dynamically
set flags=
if "%save_result_dict%"=="True" set flags=%flags% --save_result_dict
if "%save_classifier%"=="True" set flags=%flags% --save_classifier
if "%ablation%"=="True" set flags=%flags% --ablation

REM Nested loop
for %%a in (%init_seed_list%) do (
    for %%b in (%dataset_list%) do (
        for %%d in (%batchsize_list%) do (
            for %%e in (%cd_lr_list%) do (
                for %%f in (%partition_list%) do (
                    for %%g in (%epochs_classifier_list%) do (
                        for %%h in (%n_parties_list%) do (
                            for %%i in (%algorithm_list%) do (
                                for %%j in (%rounds_list%) do (
                                    for %%k in (%beta_list%) do (
                                        for %%l in (%pruneratio_list%) do (
                                            for %%m in (%model_list%) do (
                                                echo Running command: python main.py --init_seed=%%a --dataset=%%b --batch-size=%%d --cd_lr=%%e --partition=%%f --epochs_classifier=%%g --n_parties=%%h --algorithm=%%i --rounds=%%j --beta=%%k --pruneratio=%%l --model=%%m --device=%device% %flags%
                                                python main.py --init_seed=%%a --dataset=%%b --batch-size=%%d --cd_lr=%%e --partition=%%f --epochs_classifier=%%g --n_parties=%%h --algorithm=%%i --rounds=%%j --beta=%%k --pruneratio=%%l --model=%%m --device=%device% %flags%
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

endlocal

