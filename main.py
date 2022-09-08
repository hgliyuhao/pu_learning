import fairies as fa
import model
import random
from tqdm import tqdm

if __name__ == '__main__':

    for add_time in tqdm(range(5)):

        labeled_data = fa.read('train_data/labeled_data.json')
        add_data = fa.read('train_data/add_data.json')
        unlabeled_data = fa.read('train_data/unlabeled_data.json')

        train_times = 5

        random.shuffle(unlabeled_data)

        for d in add_data:
            if d[1] == 0:
                labeled_data.append(d)
        test_data = unlabeled_data[5 * (len(labeled_data)):30 *
                                   (len(labeled_data))]

        final_res = {}

        # TODO 比例平衡

        for i in range(train_times):

            datas = labeled_data + unlabeled_data[train_times * len(
                labeled_data):(train_times + 1) * len(labeled_data)]

            random.shuffle(datas)

            train_data = [d for i, d in enumerate(datas) if i % 10 != 0]
            valid_data = [d for i, d in enumerate(datas) if i % 10 == 0]

            model.train_model(train_data, valid_data,
                              'model/temp_{}.weights'.format(i))

        for i in range(train_times):

            res = model.predict(test_data, 'model/temp_{}.weights'.format(i))

            final_res[i] = res

        fa.write_json("final_res.json", final_res)

        for i, name in enumerate(test_data):

            isPos_count = 0
            isNeg_count = 0

            for t in range(train_times):
                if final_res[t][i][0] > 0.90:
                    isNeg_count += 1
                if final_res[t][i][0] < 0.05:
                    isPos_count += 1

            if isPos_count > 3:
                add_data.append([name[0], 1])

            if isNeg_count > 3:
                add_data.append([name[0], 0])
        fa.write_json('train_data/add_data.json', add_data, isIndent=True)
