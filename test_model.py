import numpy as np
import tensorflow as tf
import copy
import csv
import json

from xlam import *
from AutoML import AutoML

models_cfg = {
    1: {
        "description": "1322_top3 dataset, Single-label",
        "backup_dir": "Models/single_top3",
        "model_name": "Models/single_top3/single_top3_model.tflite",
        "labels_dict": "Models/single_top3/single_top3_dict.txt",
        "path_to_img": "Labels/",
        "json_file": "multilabeled1322_top3.json"
    },

    2: {
        "description": "1322_top5 dataset, Multi-label",
        "backup_dir": "Models/multi_top5",
        "model_name": "Models/multi_top5/multi_top5_model.tflite",
        "labels_dict": "Models/multi_top5/multi_top5_dict.txt",
        "path_to_img": "Labels/",
        "json_file": "multilabeled1322_top5.json"
    },

    3: {
        "description": "single_top5_all_imgs, Single-label",
        "backup_dir": "Models/single_top5_all_imgs",
        "model_name": "Models/single_top5_all_imgs/single_top5_all_imgs_model.tflite",
        "labels_dict": "Models/single_top5_all_imgs/single_top5_all_imgs_dict.txt",
        "path_to_img": "Labels/",
        "json_file": "multilabeled_top5_relabeled_val.json"
    },

    4: {
        "description": "top4_no12a dataset, Single-label",
        "backup_dir": "Models/single_top4_no12a",
        "model_name": "Models/single_top4_no12a/single_top4_no12a_model.tflite",
        "labels_dict": "Models/single_top4_no12a/single_top4_no12a_dict.txt",
        "path_to_img": "Labels/",
        "json_file": "multilabeled_top4_no12a.json"
    },

    5: {
        "description": "NOT WORKING binary 12a and merged top4",
        "backup_dir": "Models/binary_12a_merged",
        "model_name": "Models/binary_12a_merged/binary_12a_merged_model.tflite",
        "labels_dict": "Models/binary_12a_merged/binary_12a_merged_dict.txt",
        "path_to_img": "Labels/",
        "json_file": "multilabeled_top5.json"
    },

    6: {
        "description": "NOT WORKING binary 5ae and merged top4",
        "backup_dir": "Models/binary_5ae_merged",
        "model_name": "Models/binary_5ae_merged/binary_5ae_merged_model.tflite",
        "labels_dict": "Models/binary_5ae_merged/binary_5ae_merged_dict.txt",
        "path_to_img": "Labels/",
        "json_file": "multilabeled_top5.json"
    },

    7: {
        "description": "relabeled_val_top5 dataset, Multi-label",
        "backup_dir": "Models/multi_relabeled_top5",
        "model_name": "Models/multi_relabeled_top5/multi_top5_relabe_model.tflite",
        "labels_dict": "Models/multi_relabeled_top5/multi_top5_relabe_dict.txt",
        "path_to_img": "Labels/",
        "json_file": "multilabeled_top5_relabeled_val.json"
    },

    8: {
        "description": "relabeled_tmk_top5 dataset, Multi-label",
        "backup_dir": "Models/multi_top5_relabeled_tmk",
        "model_name": "Models/multi_top5_relabeled_tmk/multi_top5_relabeled_tmk_model.tflite",
        "labels_dict": "Models/multi_top5_relabeled_tmk/multi_top5_relabeled_tmk_dict.txt",
        "path_to_img": "Labels/",
        "json_file": "top5_relabeled_tmk_with_id.json"  # "multilabeled_top5_relabeled_tmk.json"
    }

}


def slice_and_stack(wagon_img, wagon_labels, slices_predicts, wagon_predict=None, show_cls=True):
    lbl_line_h = 80
    black_line_ = np.full((wagon_img.shape[0] + lbl_line_h, 30, 3), 0, dtype='uint8')
    stacked_slices = None

    slices_images_with_text = []

    for n, slice_id in enumerate(wagon_labels["slices"]):
        slice_label = wagon_labels["slices"][slice_id]
        slice_bbox = xcycwh_to_xyxy(wagon_img.shape[0], wagon_img.shape[1], slice_label["bbox"])
        slice_img = wagon_img[slice_bbox[1]:slice_bbox[3], slice_bbox[0]:slice_bbox[2]].copy()

        i = np.vstack((slice_img,
                       # np.full((lbl_line_h, i.shape[1], 3), 255, dtype='uint8'),
                       np.full((lbl_line_h, slice_img.shape[1], 3), 0, dtype='uint8')))

        if show_cls:
            slice_classes_str = ""
            for nn, cls in enumerate(slice_label["class"]):
                if nn + 1 == len(slice_label["class"]):
                    slice_classes_str += cls + " "
                else:
                    slice_classes_str += cls + "/"

            for topk, predict in enumerate(slices_predicts[n]):
                if topk + 1 == len(slices_predicts[n]):
                    # print(predict)
                    slice_classes_str += predict["class"] + str(predict["prob"]) + " "
                else:
                    slice_classes_str += predict["class"] + str(predict["prob"]) + "/"

            cv2.putText(i, "{}".format(slice_classes_str), (int(i.shape[1] * 0.05), int(i.shape[0] * 0.98)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, [255, 255, 255], 3)
            slices_images_with_text.append(i)
            # draw_bbox(wagon_img, slice_bbox)

        if stacked_slices is None:
            stacked_slices = i

        else:
            stacked_slices = np.hstack((stacked_slices, np.hstack((black_line_, i))))

    mid_line = np.full((lbl_line_h, stacked_slices.shape[1], 3), 0, dtype='uint8')

    if show_cls:

        if wagon_predict is not None:
            layer_str = "{} {} status: {}".format(wagon_labels["class"], wagon_predict, wagon_labels["labeled_status"])
        else:
            layer_str = "{} status: {}".format(wagon_labels["class"], wagon_labels["labeled_status"])

        cv2.putText(mid_line, layer_str, (int(mid_line.shape[1] * 0.45), int(mid_line.shape[0] * 0.8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 255, 255], 4)
    result = np.vstack((
        np.vstack((resize_image(wagon_img, width=stacked_slices.shape[1]),
                   mid_line)),
        stacked_slices))

    return result, slices_images_with_text


def test_on_val(dataset_json, model, labels, save_plots=False, multi_label=False):
    """

    @param dataset_json: Полный dataset.json В первом for loop картинки только из "val_images"
    @param model: Обученая сетка
    @param labels: Массив лейблов на которых обучалась сетка
    @param save_plots: Сохраняет графики accuracy для категории в cfg["backup_dir"]
    @param multi_label: True если сетка обучена на мульти лейбл
    """

    accuracy = {x: {"acc": 0, "values": [], "amount": 0} for x in labels}

    accuracy_roc = [{"thresh": y,
                     "below_thresh": 0,
                     "above_thresh": 0,
                     "results": {x: {"acc": 0, "values": []} for x in labels}}
                    for y in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

    layer_id = 0
    for val_sample in dataset_json["val_images"]:

        print("{}/{}".format(layer_id, len(dataset_json["val_images"])))

        image_name = os.path.split(val_sample["img"])[1]
        img = cv2.imread(os.path.join(val_dataset_path, os.path.split(val_sample["img"])[1]))

        for n, wagon_id in enumerate(val_sample["labels"]):
            wagon_labels = val_sample["labels"][wagon_id]

            wagon_cls = wagon_labels["class"]
            wagon_box_rel = wagon_labels["bbox"]
            wagon_box = xcycwh_to_xyxy(imh=img.shape[0], imw=img.shape[1], bbox=wagon_box_rel)

            wagon_img = img[wagon_box[1]:wagon_box[3], wagon_box[0]:wagon_box[2]].copy()

            # 3000x700 - best shape for 5 slices 1000x500 now
            wagon_img = resize_image(wagon_img, width=3000, height=700)
            slices_images = model.slice_wagon_img(wagon_img)
            slices_images_res = [model.preprocess_slice(x) for x in slices_images]

            slices_predicts = model.predict(batch=slices_images_res, top_k=top_k)

            for slice_id, slice_preds in enumerate(slices_predicts):

                # OneHot
                if not multi_label:
                    ground_cls = wagon_labels["slices"][str(slice_id)]["class"][0]
                    if ground_cls in labels:
                        predicted_cls, prob = slice_preds[0]["class"], slice_preds[0]["prob"]

                        accuracy[ground_cls]["amount"] += 1

                        if ground_cls == predicted_cls:
                            accuracy[ground_cls]["values"].append(True)

                            for n, i in enumerate(accuracy_roc):
                                if prob < i["thresh"]:
                                    accuracy_roc[n]["below_thresh"] += 1
                                    accuracy_roc[n]["results"][ground_cls]["values"].append(False)
                                else:
                                    accuracy_roc[n]["above_thresh"] += 1
                                    accuracy_roc[n]["results"][ground_cls]["values"].append(True)

                        else:
                            accuracy[ground_cls]["values"].append(False)

                            for n, i in enumerate(accuracy_roc):
                                accuracy_roc[n]["results"][ground_cls]["values"].append(False)
                else:
                    ground_cls = wagon_labels["slices"][str(slice_id)]["class"]
                    predicted_cls, prob = slice_preds[0]["class"], slice_preds[0]["prob"]

                    for i, cls in enumerate(ground_cls):
                        prob = False
                        if cls in labels:

                            # Подсчёт общей точности сетки для мульти лейбла
                            accuracy[cls]["amount"] += 1
                            if cls == predicted_cls:
                                accuracy[cls]["values"].append(True)
                            else:
                                accuracy[cls]["values"].append(False)

                            for prediction in slice_preds:
                                if cls == prediction["class"]:
                                    prob = prediction["prob"]
                                    for nn, ii in enumerate(accuracy_roc):
                                        if prob < ii["thresh"]:
                                            accuracy_roc[nn]["below_thresh"] += 1
                                            accuracy_roc[nn]["results"][cls]["values"].append(False)
                                        else:
                                            accuracy_roc[nn]["above_thresh"] += 1
                                            accuracy_roc[nn]["results"][cls]["values"].append(True)
                            if not prob:
                                for nn, ii in enumerate(accuracy_roc):
                                    accuracy_roc[nn]["below_thresh"] += 1
                                    accuracy_roc[nn]["results"][cls]["values"].append(False)

                    # print(ground_cls)
                    # print(predicted_cls, " | ", prob)
            layer_id += 1

    print("Test set accuracy results:")

    total_acc = {"total_acc": 0, "total_imgs": 0}

    for cls in accuracy:
        accuracy[cls]["acc"] = sum(accuracy[cls]["values"]) / len(accuracy[cls]["values"])
        total_acc["total_acc"] += accuracy[cls]["acc"] * len(accuracy[cls]["values"])
        total_acc["total_imgs"] += len(accuracy[cls]["values"])

    total_acc = int((total_acc["total_acc"] / total_acc["total_imgs"]) * 100)

    for cls in accuracy:
        print("{} = {}% ".format(
            cls, int(accuracy[cls]["acc"] * 100)))
    print("total accuracy = {}%".format(total_acc))

    # - ! - ! - ! - !

    total_acc_roc = []
    for n, i in enumerate(accuracy_roc):
        total_acc_roc.append([])
        for cls in i["results"]:
            total_acc_roc[n].extend(accuracy_roc[n]["results"][cls]["values"])
            accuracy_roc[n]["results"][cls] = int(sum(accuracy_roc[n]["results"][cls]["values"]) / len(
                accuracy_roc[n]["results"][cls]["values"]) * 100)

    for n, i in enumerate(total_acc_roc):
        total_acc_roc[n] = int(sum(total_acc_roc[n]) / len(total_acc_roc[n]) * 100)

    print("Threshold Accuracy:")
    for n, i in enumerate(total_acc_roc):
        print("threshold: {} total accuracy: {}% \nclasses accuracy: {}\n".format(
            accuracy_roc[n]["thresh"], total_acc_roc[n], accuracy_roc[n]["results"]
        ))

    if save_plots:

        import matplotlib.pyplot as plt
        plt.style.use('seaborn-whitegrid')
        category_plots = {}
        for n, i in enumerate(accuracy_roc):
            for cls in i["results"]:
                category_plots.setdefault(cls, [])
                category_plots[cls].append(i["results"][cls])
        for n, i in enumerate(category_plots):
            plt.plot([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], category_plots[i], linewidth=2, linestyle="dashed", label=i)
        plt.legend()
        plt.xlabel('threshold')
        plt.ylabel('%')
        # plt.show()
        plt.savefig(os.path.join(model.cfg["backup_dir"], 'roc_auc_category.png'))


# TODO Сделать чтоб нормальный csv делал, щас это можно просто в помойку
def create_predicts_csv(dataset_json, model):
    with open("layers_and_slices_preds.csv", 'w') as csvfile:

        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        layer_id = 0
        for val_sample in dataset_json["val_images"]:

            print("{}/{}".format(layer_id, len(dataset_json["val_images"])))

            image_name = os.path.split(val_sample["img"])[1]
            img = cv2.imread(os.path.join(val_dataset_path, os.path.split(val_sample["img"])[1]))

            for n, wagon_id in enumerate(val_sample["labels"]):
                id_ = val_sample["wagon_id"]
                wagon_labels = val_sample["labels"][wagon_id]

                wagon_cls = wagon_labels["class"]
                wagon_box_rel = wagon_labels["bbox"]
                wagon_box = xcycwh_to_xyxy(imh=img.shape[0], imw=img.shape[1], bbox=wagon_box_rel)

                wagon_img = img[wagon_box[1]:wagon_box[3], wagon_box[0]:wagon_box[2]].copy()

                # TODO запись данных для слоя, затем после для каждого слоя
                # 3000x700 - best shape for 5 slices 1000x500 now
                wagon_img = resize_image(wagon_img, width=3000, height=700)
                slices_images = model.slice_wagon_img(wagon_img)
                slices_images_res = [model.preprocess_slice(x) for x in slices_images]

                slices_predicts = model.predict(batch=slices_images_res, top_k=top_k)

                layer_predicts_sum = {}

                print_str = ""
                for slice_id, slice_preds in enumerate(slices_predicts):

                    for nn, pred_cls in enumerate(slice_preds):

                        if pred_cls["class"] not in layer_predicts_sum:
                            layer_predicts_sum[pred_cls["class"]] = 0.

                        # if pred_cls["prob"] > threshold:
                        layer_predicts_sum[pred_cls["class"]] += pred_cls["prob"]

                        if nn + 1 == len(slice_preds):
                            print_str += "{}_{}||".format(pred_cls["class"], pred_cls["prob"])
                        else:
                            print_str += "{}_{}/".format(pred_cls["class"], pred_cls["prob"])

                # TODO normalize slice predict before layer
                wagon_predict = ""
                total_prob = sum(layer_predicts_sum.values())
                for nn, i in enumerate(layer_predicts_sum):
                    if nn + 1 == len(layer_predicts_sum):
                        wagon_predict += i + "_" + str(round(layer_predicts_sum[i] / total_prob, 2))
                    else:
                        wagon_predict += i + "_" + str(round(layer_predicts_sum[i] / total_prob, 2)) + "/"

                # Формирование строки для слоя для эксель
                layer_row = [image_name, "Вагон целиком", id_]
                layer_row_wagon_clss = []
                for i in wagon_cls.split("/"):
                    layer_row_wagon_clss.append(i)
                for _ in range(3 - len(layer_row_wagon_clss)):
                    layer_row_wagon_clss.append("")
                layer_row.extend(layer_row_wagon_clss)
                for i in wagon_predict.split("/"):
                    layer_row.append(i)
                spamwriter.writerow(layer_row)

                # Формирование строк для слайсов
                for i in range(5):
                    slice_row = [image_name, "Кусок" + str(i+1), id_]
                    for cl in wagon_labels["slices"][str(i)]["class"]:
                        slice_row.append(cl)
                    for _ in range(3 - len(wagon_labels["slices"][str(i)]["class"])):
                        slice_row.append("")
                    slice_row.extend(print_str.split("||")[i].split("/"))
                    spamwriter.writerow(slice_row)

                print(print_str)


def test(dataset_json, model, top_k, show_predictions=False, layers_aggregation=False, generate_fps=False,
         save_fp_to=""):
    layer_id = 0
    # Get slices predicts

    for val_sample in dataset_json["val_images"]:
        print("{}/{}".format(layer_id, len(dataset_json["val_images"])))

        results = {"layers_ids": "", "slices_ids": [], "grounds": [[] for _ in range(top_k)],
                   "predicts": [[] for _ in range(top_k)], "probs": [[] for _ in range(top_k)]}

        image_name = os.path.split(val_sample["img"])[1]
        img = cv2.imread(os.path.join(val_dataset_path, os.path.split(val_sample["img"])[1]))

        for n, wagon_id in enumerate(val_sample["labels"]):
            wagon_labels = val_sample["labels"][wagon_id]

            wagon_cls = wagon_labels["class"]
            wagon_box_rel = wagon_labels["bbox"]
            wagon_box = xcycwh_to_xyxy(imh=img.shape[0], imw=img.shape[1], bbox=wagon_box_rel)

            wagon_img = img[wagon_box[1]:wagon_box[3], wagon_box[0]:wagon_box[2]].copy()

            # 3000x700 - best shape for 5 slices 1000x500 now
            wagon_img = resize_image(wagon_img, width=3000, height=700)
            slices_images = model.slice_wagon_img(wagon_img)
            slices_images_res = [model.preprocess_slice(x) for x in slices_images]

            slices_predicts = model.predict(batch=np.array(slices_images_res), top_k=top_k)

            if show_predictions:
                stacked, slices_images_with_text = slice_and_stack(wagon_img, wagon_labels, slices_predicts)
                print_str = ""
                for slice_preds in slices_predicts:
                    for nn, pred_cls in enumerate(slice_preds):
                        if nn + 1 == len(slice_preds):
                            print_str += "{}_{} || ".format(pred_cls["class"], pred_cls["prob"])
                        else:
                            print_str += "{}_{}/".format(pred_cls["class"], pred_cls["prob"])
                print(print_str)
                show_image(stacked)

            if layers_aggregation:
                layer_predicts_sum = {}

                print_str = ""
                for slice_id, slice_preds in enumerate(slices_predicts):

                    ground_cls = wagon_labels["slices"][str(slice_id)]["class"][0]

                    for nn, pred_cls in enumerate(slice_preds):

                        if pred_cls["class"] not in layer_predicts_sum:
                            layer_predicts_sum[pred_cls["class"]] = 0.

                        # if pred_cls["prob"] > threshold:
                        layer_predicts_sum[pred_cls["class"]] += pred_cls["prob"]

                        if nn + 1 == len(slice_preds):
                            print_str += "{}_{} || ".format(pred_cls["class"], pred_cls["prob"])
                        else:
                            print_str += "{}_{}/".format(pred_cls["class"], pred_cls["prob"])

                # TODO normalize slice predict before layer
                wagon_predict = ""
                total_prob = sum(layer_predicts_sum.values())
                for nn, i in enumerate(layer_predicts_sum):
                    if nn + 1 == len(layer_predicts_sum):
                        wagon_predict += i + str(round(layer_predicts_sum[i] / total_prob, 2))
                    else:
                        wagon_predict += i + str(round(layer_predicts_sum[i] / total_prob, 2)) + "/"

                stacked, slices_images_with_text = slice_and_stack(wagon_img=wagon_img, wagon_labels=wagon_labels,
                                                                   wagon_predict=wagon_predict,
                                                                   slices_predicts=slices_predicts)

                print(print_str)
                show_image(stacked)

            if generate_fps:
                results["layers_ids"] = layer_id
                results["slices_ids"].append([])
                for i in range(top_k):
                    results["grounds"][i].append([])
                    results["predicts"][i].append([])
                    results["probs"][i].append([])

                for n_, slice_id in enumerate(wagon_labels["slices"]):
                    slice_classes_list = copy.deepcopy(wagon_labels["slices"][slice_id]["class"])
                    results["slices_ids"][0].append(int(slice_id))
                    while len(slice_classes_list) < top_k:
                        slice_classes_list.append(None)

                    for nn, cls in enumerate(slice_classes_list):
                        results["grounds"][nn][0].append(cls)

                stacked, slices_images_with_text = slice_and_stack(wagon_img, wagon_labels, slices_predicts)

                print_str = ""
                for slice_id, slice_preds in enumerate(slices_predicts):

                    ground_cls = wagon_labels["slices"][str(slice_id)]["class"][0]

                    for nn, pred_cls in enumerate(slice_preds):

                        results["predicts"][nn][0].append(pred_cls["class"])
                        results["probs"][nn][0].append(pred_cls["prob"])

                        if nn == 0 and pred_cls["class"] != ground_cls:

                            layer_save_dir = os.path.join(save_fp_to, "fp", ground_cls, "layers")
                            slice_save_dir = os.path.join(save_fp_to, "fp", ground_cls, "slices")

                            if not os.path.isdir(layer_save_dir):
                                os.makedirs(layer_save_dir)

                            if not os.path.isdir(slice_save_dir):
                                os.makedirs(slice_save_dir)

                            layer_save_name = os.path.join(layer_save_dir, "{}.jpg".format(layer_id))
                            slice_save_name = os.path.join(slice_save_dir, "{}_{}.jpg".format(layer_id, slice_id))

                            if not os.path.isfile(layer_save_name):
                                cv2.imwrite(layer_save_name, stacked)

                            cv2.imwrite(slice_save_name, slices_images_with_text[slice_id])

                        if nn + 1 == len(slice_preds):
                            print_str += "{}_{} || ".format(pred_cls["class"], pred_cls["prob"])
                        else:
                            print_str += "{}_{}/".format(pred_cls["class"], pred_cls["prob"])

        layer_id += 1


if __name__ == "__main__":

    input_cfg = 8

    test_on_validatoin = False
    create_csv = True

    path_to_img = models_cfg[input_cfg]["path_to_img"]
    with open(os.path.join(path_to_img, models_cfg[input_cfg]["json_file"]), "r") as f:
        dataset_json = json.load(f)

    # TODO переменная захардкодена для всех функций. Сделать нормально.
    val_dataset_path = "Labels/val_dataset"

    model = AutoML(cfg=models_cfg[input_cfg])
    top_k = 3

    # TODO оформмить всё в функцию которая на вход принимет значения generate_fps, show_predictions, create_csvfile
    # test(dataset_json, model, top_k,
    #      show_predictions=False,
    #      layers_aggregation=True,
    #      generate_fps=False, save_fp_to=models_cfg[input_cfg]["backup_dir"])

    if test_on_validatoin:
        test_on_val(dataset_json, model, model.labels, multi_label=False, save_plots=False)

    if create_csv:
        create_predicts_csv(dataset_json, model)
