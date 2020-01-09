import logging

import click
import numpy as np

import settings
import task
import utils


@settings.main.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
@click.option('-t', '--task', default='UserEmbedding')
@click.option('-a', '--arch', default='avg')
@click.option('-r', '--round', default=6)
@click.option('-y', '--days', default=30)
@click.option('-e', '--epochs', default=10)
@click.option('-b', '--batch-size', default=100)
@click.option('-s', '--training-step', default=10000)
@click.option('-v', '--validation-step', default=1000)
@click.option('-cvi', '--validation-impression', default=1000)
@click.option('-i', '--testing-impression', default=1000)
@click.option('-l', '--learning-rate', default=0.001)
@click.option('-cld', '--learning-rate-decay', default=0.2)
@click.option('-g', '--gain', default=1.0)
@click.option('-w', '--window-size', default=10)
@click.option('-d', '--dropout', default=0.2)
@click.option('-n', '--negative-samples', default=4)
@click.option('-h', '--hidden-dim', default=400)
@click.option('-nn', '--nonlocal-negative-samples', default=0)
@click.option('-ceb', '--enable-baseline', is_flag=True)
@click.option('--title-filter-shape', default=(400, 3), nargs=2)
@click.option('--title-shape', default=20)
@click.option('--body-shape', default=200)
@click.option('--user-embedding-dim', default=200)
@click.option('--textual-embedding-dim', default=300)
@click.option('--textual-embedding-trainable', is_flag=True)
@click.option('--debug', is_flag=True)
@click.option('--background', is_flag=True)
@click.option('--name', default='')
@click.option('-p', '--pretrain-name', default='')
@click.option('-cep', '--enable-pretrain-encoder', is_flag=True)
@click.option('-cpt', '--pretrain-encoder-trainable', is_flag=True)
@click.option('--personal-embedding-dim', default=20)
@click.option('--news-encoder', default='cnnatt')
@click.option('--score-model', default='dot')
@click.option('--body-sent-cnt', default=50)
@click.option('--body-sent-len', default=30)
@click.option('--body-filter-shape', default=(400, 3), nargs=2)
@click.option('--max-impression', default=200)
@click.option('--max-impression-pos', default=7)
@click.option('--max-impression-neg', default=200)
@click.option('--test-window-size', default=100)
@click.option('--vertical-embedding-dim', default=10)
@click.option('--subvertical-embedding-dim', default=20)
@settings.pass_config
def train(config: settings.Config):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(config.log_output),
            logging.StreamHandler()
        ]
    )

    task_handler = task.get(config)

    training_data = task_handler.train

    for epoch in range(config.epochs):
        logging.info('[+] start epoch {}'.format(epoch))
        model = task_handler.build_model(epoch)
        history = model.fit_generator(
            training_data,
            task_handler.training_step,
            epochs=epoch + 1,
            initial_epoch=epoch,
            verbose=1 if config.debug and not config.background else 2)
        utils.logging_history(history)
        if hasattr(task_handler, 'callback'):
            task_handler.callback(epoch)
        try:
            evaluations = model.evaluate_generator(
                task_handler.valid,
                task_handler.validation_step,
                verbose=1 if config.debug and not config.background else 2)
            utils.logging_evaluation(dict(zip(model.metrics_names, evaluations)))
        except:
            pass
        if hasattr(task_handler, 'callback_valid'):
            task_handler.callback_valid(epoch)
        logging.info('[-] finish epoch {}'.format(epoch))

    task_handler.save_model()

    return 0


@settings.main.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
@click.option('-t', '--task', default='Cook')
@click.option('-a', '--arch', default='avg')
@click.option('-r', '--round', default=6)
@click.option('-y', '--days', default=30)
@click.option('-e', '--epochs', default=10)
@click.option('-b', '--batch-size', default=100)
@click.option('-s', '--training-step', default=10000)
@click.option('-v', '--validation-step', default=1000)
@click.option('-cvi', '--validation-impression', default=1000)
@click.option('-i', '--testing-impression', default=1000)
@click.option('-l', '--learning-rate', default=0.001)
@click.option('-cld', '--learning-rate-decay', default=0.2)
@click.option('-g', '--gain', default=1.0)
@click.option('-w', '--window-size', default=10)
@click.option('-d', '--dropout', default=0.2)
@click.option('-n', '--negative-samples', default=4)
@click.option('-h', '--hidden-dim', default=400)
@click.option('-nn', '--nonlocal-negative-samples', default=0)
@click.option('-ceb', '--enable-baseline', is_flag=True)
@click.option('--title-filter-shape', default=(400, 3), nargs=2)
@click.option('--title-shape', default=20)
@click.option('--body-shape', default=200)
@click.option('--user-embedding-dim', default=200)
@click.option('--textual-embedding-dim', default=300)
@click.option('--textual-embedding-trainable', is_flag=True)
@click.option('--debug', is_flag=True)
@click.option('--background', is_flag=True)
@click.option('--name', default='')
@click.option('-p', '--pretrain-name', default='')
@click.option('-cep', '--enable-pretrain-encoder', is_flag=True)
@click.option('-cpt', '--pretrain-encoder-trainable', is_flag=True)
@click.option('--personal-embedding-dim', default=20)
@click.option('--news-encoder', default='cnnatt')
@click.option('--score-model', default='ddot')
@click.option('--id-keep', default=1.0)
@click.option('--body-sent-cnt', default=50)
@click.option('--body-sent-len', default=30)
@click.option('--body-filter-shape', default=(400, 3), nargs=2)
@click.option('--max-impression', default=200)
@click.option('--max-impression-pos', default=7)
@click.option('--max-impression-neg', default=200)
@click.option('--test-window-size', default=100)
@click.option('--vertical-embedding-dim', default=15)
@click.option('--subvertical-embedding-dim', default=35)
@click.option('--use-vertical', is_flag=True)
@click.option('--use-vertical-type', default='vs')
@click.option('--use-generator', is_flag=True)
@click.option('-coe', '--lrd-on-epochs', cls=settings.PythonLiteralOption, default=[1, 3])
@settings.pass_config
def cook(config: settings.Config):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(config.log_output),
            logging.StreamHandler()
        ]
    )

    task_handler = task.get(config)

    for epoch in range(config.epochs):
        logging.info('[+] start epoch {}'.format(epoch))
        model = task_handler.build_model(epoch)
        if config.use_generator:
            history = model.fit_generator(
                task_handler.train(),
                task_handler.training_step,
                epochs=epoch + 1,
                initial_epoch=epoch,
                verbose=1 if config.debug and not config.background else 2)
        else:
            history = model.fit(
                *task_handler.train(),
                config.batch_size,
                epochs=epoch + 1,
                initial_epoch=epoch,
                shuffle=True,
                verbose=1 if config.debug and not config.background else 2)
        utils.logging_history(history)
        if hasattr(task_handler, 'callback'):
            task_handler.callback(epoch)
        try:
            if config.use_generator:
                evaluations = model.evaluate_generator(
                    task_handler.valid(),
                    steps=task_handler.validation_step,
                    verbose=1 if config.debug and not config.background else 2)
            else:
                evaluations = task_handler.test_model.evaluate(
                    *task_handler.valid(),
                    config.batch_size,
                    verbose=1 if config.debug and not config.background else 0)
            utils.logging_evaluation(dict(zip(task_handler.test_model.metrics_names, evaluations)))
        except Exception as e:
            print(e)
        if hasattr(task_handler, 'callback_valid'):
            task_handler.callback_valid(epoch)
        logging.info('[-] finish epoch {}'.format(epoch))

    if config.use_generator:
        users_ = []
        imprs_ = []
        mask_ = []
        y_true_ = []
        y_pred_ = []
        for feature, [users, imprs, mask, y_true] in task_handler.test():
            users_.append(users)
            imprs_.append(imprs)
            mask_.append(mask)
            y_true_.append(y_true)
            y_pred_.append(task_handler.test_model.predict_on_batch(feature).reshape((-1,)))
        users = np.hstack(users_)
        imprs = np.hstack(imprs_)
        mask = np.hstack(mask_)
        y_true = np.hstack(y_true_)
        y_pred = np.hstack(y_pred_)
    else:
        feature, [users, imprs, mask, y_true] = task_handler.test()
        y_pred = task_handler.test_model.predict(
            feature,
            batch_size=config.batch_size,
            verbose=1 if config.debug and not config.background else 0).reshape((-1,))

    class Result:
        def __init__(self, auc, mrr, ndcgv, ndcgx, pos, size, idx):
            self.auc = auc
            self.mrr = mrr
            self.ndcgv = ndcgv
            self.ndcgx = ndcgx
            self.pos = pos
            self.size = size
            self.idx = idx

        @property
        def result(self):
            return dict(auc=self.auc, ndcgx=self.ndcgx, ndcgv=self.ndcgv, mrr=self.mrr)

        @property
        def info(self):
            return dict(pos=self.pos, size=self.size, num=self.idx * 2 + 1)

    def average(results):
        return Result(
            np.mean([result.auc for result in results]),
            np.mean([result.mrr for result in results]),
            np.mean([result.ndcgv for result in results]),
            np.mean([result.ndcgx for result in results]),
            np.mean([result.pos for result in results]),
            np.mean([result.size for result in results]),
            np.mean([result.idx for result in results]))

    present_user = users[0]
    present_impr = imprs[0]
    index = 0
    impr_index = 0
    user_results = []
    impr_results = []
    iv_user_results = []
    oov_user_results = []
    for i, user, impr in zip(range(1, len(y_pred)), users[1:], imprs[1:]):
        if user != present_user or impr != present_impr:
            try:
                impr_results.append(Result(
                    task.roc_auc_score(y_true[index:i], y_pred[index:i]),
                    utils.mrr_score(y_true[index:i], y_pred[index:i]),
                    utils.ndcg_score(y_true[index:i], y_pred[index:i], 5),
                    utils.ndcg_score(y_true[index:i], y_pred[index:i], 10),
                    sum(y_true[index:i]),
                    i - index,
                    len(impr_results)))
                index = i
                present_impr = impr
            except Exception as e:
                utils.interactive_console(locals())
        if user != present_user:
            avg = average(impr_results[impr_index:])
            user_results.append(avg)
            if mask[index] == 1:
                iv_user_results.append(avg)
            elif mask[index] == 0:
                oov_user_results.append(avg)
            impr_index = len(impr_results)
            present_user = user

    user_result = average(user_results)
    impr_result = average(impr_results)
    iv_user_result = average(iv_user_results)
    oov_user_result = average(oov_user_results)

    utils.logging_evaluation(user_result.result)
    utils.logging_evaluation(user_result.info)
    utils.logging_evaluation(impr_result.result)
    utils.logging_evaluation(impr_result.info)
    utils.logging_evaluation(iv_user_result.result)
    utils.logging_evaluation(iv_user_result.info)
    utils.logging_evaluation(oov_user_result.result)
    utils.logging_evaluation(oov_user_result.info)
    return 0


@settings.main.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
@click.option('-t', '--task', default='RunUserModel')
@click.option('-a', '--arch', default='catt')
@click.option('-e', '--epochs', default=10)
@click.option('-b', '--batch-size', default=100)
@click.option('-s', '--training-step', default=10000)
@click.option('-v', '--validation-step', default=1000)
@click.option('-i', '--testing-impression', default=1000)
@click.option('-l', '--learning-rate', default=0.001)
@click.option('-cld', '--learning-rate-decay', default=0.2)
@click.option('-g', '--gain', default=1.0)
@click.option('-w', '--window-size', default=10)
@click.option('-d', '--dropout', default=0.2)
@click.option('-n', '--negative-samples', default=4)
@click.option('-h', '--hidden-dim', default=200)
@click.option('-nn', '--nonlocal-negative-samples', default=0)
@click.option('-ceb', '--enable-baseline', is_flag=True)
@click.option('--title-filter-shape', default=(400, 3), nargs=2)
@click.option('--title-shape', default=20)
@click.option('--body-shape', default=200)
@click.option('--user-embedding-dim', default=200)
@click.option('--textual-embedding-dim', default=300)
@click.option('--textual-embedding-trainable', is_flag=True)
@click.option('--debug', is_flag=True)
@click.option('--background', is_flag=True)
@click.option('--name', default='')
@click.option('-p', '--pretrain-name', default='')
@click.option('-cep', '--enable-pretrain-encoder', is_flag=True)
@click.option('-cpt', '--pretrain-encoder-trainable', is_flag=True)
@settings.pass_config
def users(config: settings.Config):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(config.log_output),
            logging.StreamHandler()
        ]
    )

    task_handler = task.get(config)
    task_handler.save_result()
    return 0


@settings.main.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
@click.option('-t', '--task', default='test')
@click.option('-b', '--batch-size', default=32)
@click.option('--debug', is_flag=True)
@click.option('--name', default='')
@settings.pass_config
def score(config: settings.Config):
    task_handler = task.get(config)
    with utils.open(next(task_handler), 'w') as file:
        model = next(task_handler)
        for batch_info, batch_data in task_handler:
            batch_pred = model.predict_on_batch(batch_data)
            for (session, label, score), pred in zip(batch_info, batch_pred):
                file.write('{}\t{}\t{}\t{}\n'.format(session, label, score, pred[0]))

    return 0


@settings.main.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
@click.option('--name', default='')
@click.option('--debug', is_flag=True)
@settings.pass_config
def evaluate(config: settings.Config):
    import numpy as np
    import pandas as pd
    import sklearn.metrics as metrics

    def dcg_score(y_true, y_score, k=10):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def ndcg_score(y_true, y_score, k=10):
        best = dcg_score(y_true, y_true, k)
        actual = dcg_score(y_true, y_score, k)
        return actual / best

    def mrr_score(y_true, y_score):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order)
        rr_score = y_true / (np.arange(len(y_true)) + 1)
        return np.sum(rr_score) / np.sum(y_true)

    with utils.open(config.result_input) as file:
        df = pd.read_csv(file, sep='\t', names=['session', 'label', 'baseline', 'pred1'])
    df['pred2'] = (df.baseline + df.pred1) / 2

    auc0 = metrics.roc_auc_score(df.label.values, df.baseline.values)
    auc1 = metrics.roc_auc_score(df.label.values, df.pred1.values)
    auc2 = metrics.roc_auc_score(df.label.values, df.pred2.values)
    print('auc:', auc0, auc1, auc2)

    dfs = df.groupby('session').agg(list)
    print(dfs.count())
    ndcgx0 = np.mean(dfs.apply(lambda x: ndcg_score(x.label, x.baseline, 10), axis=1))
    ndcgx1 = np.mean(dfs.apply(lambda x: ndcg_score(x.label, x.pred1, 10), axis=1))
    ndcgx2 = np.mean(dfs.apply(lambda x: ndcg_score(x.label, x.pred2, 10), axis=1))
    print('ndcg@10:', ndcgx0, ndcgx1, ndcgx2)

    ndcgv0 = np.mean(dfs.apply(lambda x: ndcg_score(x.label, x.baseline, 5), axis=1))
    ndcgv1 = np.mean(dfs.apply(lambda x: ndcg_score(x.label, x.pred1, 5), axis=1))
    ndcgv2 = np.mean(dfs.apply(lambda x: ndcg_score(x.label, x.pred2, 5), axis=1))
    print('ndcg@5:', ndcgv0, ndcgv1, ndcgv2)

    mrr0 = np.mean(dfs.apply(lambda x: mrr_score(x.label, x.baseline), axis=1))
    mrr1 = np.mean(dfs.apply(lambda x: mrr_score(x.label, x.pred1), axis=1))
    mrr2 = np.mean(dfs.apply(lambda x: mrr_score(x.label, x.pred2), axis=1))
    print('mrr:', mrr0, mrr1, mrr2)

    print('avg events in session:', df.count().label / dfs.count().label)

    mask = df.label == 1
    print('avg positives in session:', df.label[mask].count() / dfs.count().label)

    return 0


@settings.main.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
@click.option('-t', '--task', default='TestPipeline')
@click.option('-b', '--batch-size', default=256)
@click.option('--name', default='')
@click.option('--pipeline-input', default='')
@settings.pass_config
def pipeline(config: settings.Config):
    task_handler = task.get(config)
    task_handler.load_model()
    task_handler.test_doc_vec()
    task_handler.test_user_vec()
    task_handler.test_user_doc_score()
    task_handler.test_correct()
    return 0


if __name__ == '__main__':
    exit(settings.main(auto_envvar_prefix='MSN_NEWS'))
