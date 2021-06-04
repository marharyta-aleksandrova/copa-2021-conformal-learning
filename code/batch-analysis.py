from scripts import *
import time
import datetime


def write_message(message, log_file):
    print(message)
    log_file.write(message + '\n')
    log_file.flush()
    pass


def run(path_to_res, seed=1, func=analysis_with_3_func):
    datasets_load = {
        'balance': load_balance, 'cars': load_cars, 'ecoli': load_ecoli, 'glass': load_glass,
        'iris': load_iris, 'user': load_user, 'wave': load_wave, 'wine': load_wine,
        'wine_Red': load_wine_red, 'wine_White': load_wine_white, 'yeast': load_yeast, 'zoo': load_zoo
    }
    data_order = [
        'balance',
        'glass',
        'user',
        'wine_Red',
        'wine_White',
        'yeast',
        'iris',
        'cars',
        # 'zoo',
        # 'ecoli',
        # 'wave',
        # 'wine',
    ]

    with open(path_to_res + 'log/A_info_{}.log'.format(seed), 'w') as log_file:
        message = 'Started {}'.format(datetime.datetime.now())
        write_message(message, log_file)

        for i in range(0, len(data_order)):
            dataset_name = data_order[i]
            data = datasets_load[dataset_name]()
            data_df = pd.DataFrame(data['data'])
            data_df['target'] = data['target']
            tmp_df = data_df.groupby('target').count()[0]
            message = 'Dataset: {}, classes: {}, inst: {}, attributes: {}; max_per_class = {}, min_per_class = {}' \
                .format(dataset_name, len(data['target_names']), len(data_df), len(data['data'][0]), tmp_df.max(),
                        tmp_df.min())
            write_message(message, log_file)

            # run classifiers on this dataset & measure time
            start = time.time()
            svm_mean_df = func(data=data, model_obj=svm_model_obj,
                               params=svm_params, epsilon_arr=eps_err,
                               smoothing=False, seed=seed, sub_verbose=False)
            end = time.time()
            write_message('\tSVM time: {}'.format(end - start), log_file)

            start = time.time()
            dt_mean_df = func(data=data, model_obj=dt_model_obj,
                              params={MIN_SAMPLES_KEY: max(5, int(
                                  dt_min_sample_ratio * len(data['data'])))},
                              epsilon_arr=eps_err,
                              smoothing=False, seed=seed, sub_verbose=False)
            end = time.time()
            write_message('\tDT  time: {}'.format(end - start), log_file)

            start = time.time()
            knn_mean_df = func(data=data, model_obj=k_kneighbors_model_obj,
                               params=k_kneighbors_params,
                               epsilon_arr=eps_err,
                               smoothing=False, seed=seed, sub_verbose=False)
            end = time.time()
            write_message('\tKNN time: {}'.format(end - start), log_file)

            start = time.time()
            ada_boost_mean_df = func(data=data, model_obj=ada_boost_model_obj,
                                     params=ada_boost_params,
                                     epsilon_arr=eps_err,
                                     smoothing=False, seed=seed, sub_verbose=False)
            end = time.time()
            write_message('\tADA time: {}'.format(end - start), log_file)

            start = time.time()
            gussian_nb_mean_df = func(data=data, model_obj=gaussian_nb_model_obj,
                                      params=gaussian_nb_params, epsilon_arr=eps_err,
                                      smoothing=False, seed=seed, sub_verbose=False)
            end = time.time()
            write_message('\tGNB time: {}'.format(end - start), log_file)

            start = time.time()
            mlp_classifier_mean_df = func(data=data, model_obj=mlp_classifier_model_obj,
                                          params=mlp_classifier_params, epsilon_arr=eps_err,
                                          smoothing=False, seed=seed, sub_verbose=False)
            end = time.time()
            write_message('\tMLP time: {}'.format(end - start), log_file)

            random_forest_params[MIN_SAMPLES_KEY] = max(5, int(dt_min_sample_ratio * len(data['data'])))
            start = time.time()
            random_forest_mean_df = func(data=data, model_obj=random_forest_model_obj,
                                         params=random_forest_params, epsilon_arr=eps_err,
                                         smoothing=False, seed=seed, sub_verbose=False)
            end = time.time()
            write_message('\tRF  time: {}'.format(end - start), log_file)
            write_message('\t\tRF params: {}'.format(random_forest_params), log_file)

            """
            start = time.time()
            gaussian_process_mean_df = pd.DataFrame({})
            try:
                gaussian_process_mean_df = func(data=data, model_obj=gaussian_process_obj,
                                                                params=gaussian_process_params, epsilon_arr=eps_err,
                                                                smoothing=False, seed=seed, sub_verbose=False)
            except:
                pass
            end = time.time()
            write_message('\tGPr time: {}'.format(end - start), log_file)
            """

            start = time.time()
            quadratic_discriminant_mean_df = pd.DataFrame({})
            try:
                quadratic_discriminant_mean_df = func(data=data,
                                                      model_obj=quadratic_discriminant_model_obj,
                                                      params=quadratic_discriminant_params,
                                                      epsilon_arr=eps_err,
                                                      smoothing=False, seed=seed, sub_verbose=False)
            except:
                pass
            end = time.time()
            write_message('\tQDA time: {}'.format(end - start), log_file)

            # give names to dataFrames
            svm_mean_df.name = 'SVM'
            dt_mean_df.name = 'DT'
            knn_mean_df.name = 'KNN'
            ada_boost_mean_df.name = 'Ada'
            gussian_nb_mean_df.name = 'GNB'
            mlp_classifier_mean_df.name = 'MPR'
            random_forest_mean_df.name = 'RF'
            # gaussian_process_mean_df.name = 'GPr.'
            quadratic_discriminant_mean_df.name = 'QDA'

            all_results = [
                svm_mean_df,
                dt_mean_df,
                knn_mean_df,
                ada_boost_mean_df,
                gussian_nb_mean_df,
                mlp_classifier_mean_df,
                random_forest_mean_df,
                # gaussian_process_mean_df,
                quadratic_discriminant_mean_df
            ]

            for df in all_results:
                file_name = path_to_res + 'experimental-results/' + dataset_name + '_' + df.name + '_' + str(
                    seed) + '.csv'
                df.to_csv(file_name)
                pass
            pass

        message = 'Finished {}'.format(datetime.datetime.now())
        write_message(message, log_file)
        pass

    pass


if __name__ == '__main__':
    for i in range(0, 11):
        run(path_to_res='analysis-results/', seed=i, func=analysis_with_3_func_in_1)

    pass
