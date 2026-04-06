import numpy as np
import pandas as pd
from enum import Enum
import base64
import zlib
import requests
import re
import json
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt

requests.packages.urllib3.util.connection.HAS_IPV6 = False

with open("objectIDTable.json", 'r') as f:
    object_id_table: dict[str, str] = json.load(f)

class Difficulty(Enum):
    Easy = 2
    Normal = 3
    Hard = 4
    Harder = 5
    Insane = 6
    Demon = 8


class Object:
    type: str
    x: float
    y: float
    portal: str | None

    def __init__(self, obj_id: str, x: str, y: str):
        self.type = object_id_table[obj_id]
        self.x = float(x)
        self.y = float(y)

        if obj_id == '12':
            self.portal = 'cube'
        elif obj_id == '13':
            self.portal = 'ship'
        elif obj_id == '47':
            self.portal = 'ball'
        elif obj_id == '111':
            self.portal = 'ufo'
        elif obj_id == '660':
            self.portal = 'wave'
        elif obj_id == '286' or obj_id == '287':
            self.portal = 'dual'
        elif obj_id == '10' or obj_id == '11':
            self.portal = 'gravity'
        elif obj_id == '45' or obj_id == '46':
            self.portal = 'mirror'
        elif obj_id == '99' or obj_id == '101':
            self.portal = 'size'
        elif obj_id == '200':
            self.portal = 'speed-half'
        elif obj_id == '201':
            self.portal = 'speed-one'
        elif obj_id == '202':
            self.portal = 'speed-two'
        elif obj_id == '203':
            self.portal = 'speed-three'
        else:
            self.portal = None


class Property:
    gamemode: int
    speed: int
    is_mirror: bool
    is_flipped: bool
    is_mini: bool
    is_dual: bool

    def __init__(self, raw_property: dict[str, str]):
        self.gamemode = int(raw_property['kA2']) if 'kA2' in raw_property  else 0
        self.speed = int(raw_property['kA4']) if 'kA4' in raw_property  else 0
        self.is_mini = raw_property['kA3'] == '1' if 'kA3' in raw_property  else False
        self.is_dual = raw_property['kA8'] == '1' if 'kA8' in raw_property  else False
        self.is_flipped = raw_property['kA11'] == '1' if 'kA11' in raw_property  else False
        self.is_mirror = raw_property['kA28'] == '1' if 'kA28' in raw_property  else False


def GetList(difficulty: Difficulty) -> list[int]:
    url = "https://history.geometrydash.eu/api/v1/search/level/advanced/"
    params = {
        "limit": 200,
        "sort": "cache_likes:desc",
        "filter": f"cache_game_version <= 19 AND cache_filter_difficulty = {difficulty.value} AND cache_stars > 1 AND cache_level_string_available = true AND cache_two_player = false"
    }
    response = requests.get(url, params).json()
    level_list_raw: list = response['hits']

    level_list = []
    for lvl in level_list_raw:
        level_list.append(int(lvl['online_id']))

    return level_list


def GetLevel(level_id: int) -> dict | None:
    url = f"https://history.geometrydash.eu/api/v1/level/{level_id}/"

    response = requests.get(url).json()
    records: list[dict] = response['records']
    for item in reversed(records):
        if item['level_string_available']:
            url_gmd = f"https://history.geometrydash.eu/level/{level_id}/{item['id']}/download/"

            gmd_string = requests.get(url_gmd).text
            pattern = r"<k>(.*?)</k>\s*<(i|s)>(.*?)</\2>"
            matches = re.findall(pattern, gmd_string, re.DOTALL)

            gmd_raw = {}
            for key, tag_type, value in matches:
                if tag_type == 'i':
                    gmd_raw[key] = int(value)
                else:
                    gmd_raw[key] = value

            gmd = {
                "ID": gmd_raw['k1'],
                "Name": gmd_raw['k2'],
                "Data": Decode(gmd_raw['k4'])
            }

            if gmd["Data"] is None:
                return None

            return gmd

    raise FileNotFoundError("File not found.")


def Decode(raw_string: str):
    try:
        base64_decoded = base64.urlsafe_b64decode(raw_string.encode())
        decompressed = zlib.decompress(base64_decoded, 15 | 32)
        data_string = decompressed.decode()
    except:
        return FormatLevelData(raw_string)
    return FormatLevelData(data_string)


def ParseKVPairs(comma_sep_string: str):
    parts = comma_sep_string.split(',')
    return {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1, 2)}


def FormatLevelData(level_string: str):
    segments = level_string.split(';')
    level_properties = ParseKVPairs(segments[0])
    level_objects = []
    for seg in segments[1:]:
        if seg.strip():
            level_objects.append(ParseKVPairs(seg))

    properties = Property(level_properties)
    objects = [Object(obj['1'], obj['2'], obj['3']) for obj in level_objects if '1' in obj and obj['1'] in object_id_table and '2' in obj and '3' in obj]
    return properties, objects


def GetDataset():
    dataset: dict[int, list[dict]] = {}
    for diff in Difficulty:
        print(f"Fetching levels with difficulty '{diff.name}'")
        ids = GetList(diff)
        print(f"Found {len(ids)} matches (max: 200)")
        count = 0
        for lvl_id in ids:
            if count % 10 == 0:
                print(f"Fetching level data ({count}/{len(ids)})")
            count += 1

            if diff.value not in dataset:
                dataset[diff.value] = []
            lvl_data = GetLevel(lvl_id)
            if lvl_data is not None:
                dataset[diff.value].append(lvl_data)
        print("")

    print("Dataset created successfully.")
    return dataset


def ExtractFeatures(dataset: dict[int, list[dict]], segment_size: int = 10):
    data = {
        "id": [],
        "name": [],
        "difficulty": [],
        "x-length": [],
        "y-range": [],
        "portal-cube": [],
        "portal-ship": [],
        "portal-ball": [],
        "portal-ufo": [],
        "portal-wave": [],
        "portal-gravity": [],
        "portal-mirror": [],
        "portal-size": [],
        "portal-dual": [],
        "portal-speed": []
    }

    for difficulty in dataset:
        for level in dataset[difficulty]:
            #level_property: Property = level["Data"][0]
            level_objects: list[Object] = sorted(level["Data"][1], key=lambda obj: obj.x)
            if len(level_objects) == 0:
                continue

            level_length = max(level_objects, key=lambda obj: obj.x).x

            # global features
            data["id"].append(level["ID"])
            data["name"].append(level["Name"])
            data["difficulty"].append(Difficulty(difficulty).name)
            data["x-length"].append(level_length)
            data["y-range"].append(max(level_objects, key=lambda obj: obj.y).y)

            portal_counts = Counter(obj.portal for obj in level_objects)
            data["portal-cube"].append(portal_counts["cube"])
            data["portal-ship"].append(portal_counts["ship"])
            data["portal-ball"].append(portal_counts["ball"])
            data["portal-ufo"].append(portal_counts["ufo"])
            data["portal-wave"].append(portal_counts["wave"])
            data["portal-gravity"].append(portal_counts["gravity"])
            data["portal-mirror"].append(portal_counts["mirror"])
            data["portal-size"].append(portal_counts["size"])
            data["portal-dual"].append(portal_counts["dual"])
            data["portal-speed"].append(portal_counts["speed-half"] + portal_counts["speed-one"] + portal_counts["speed-two"] + portal_counts["speed-three"])

            # segment features
            segment_len = level_length / segment_size
            for i in range(segment_size):
                segment = [obj for obj in level_objects if segment_len * i < obj.x <= segment_len * (i + 1)]
                obj_type_counts = Counter(obj.type for obj in segment)
                obj_type_densities = {obj: obj_type_counts[obj] / segment_len for obj in obj_type_counts}
                if f"segment{i}-density-spike" not in data:
                    data[f"segment{i}-density-block"] = []
                    data[f"segment{i}-density-spike"] = []
                    data[f"segment{i}-density-slope"] = []
                    data[f"segment{i}-density-saw"] = []
                    data[f"segment{i}-density-portal"] = []
                    data[f"segment{i}-density-pad"] = []
                    data[f"segment{i}-density-orb"] = []
                data[f"segment{i}-density-block"].append(obj_type_densities["block"] if "block" in obj_type_densities else 0)
                data[f"segment{i}-density-spike"].append(obj_type_densities["spike"] if "spike" in obj_type_densities else 0)
                data[f"segment{i}-density-slope"].append(obj_type_densities["slope"] if "slope" in obj_type_densities else 0)
                data[f"segment{i}-density-saw"].append(obj_type_densities["saw"] if "saw" in obj_type_densities else 0)
                data[f"segment{i}-density-portal"].append(obj_type_densities["portal"] if "portal" in obj_type_densities else 0)
                data[f"segment{i}-density-pad"].append(obj_type_densities["pad"] if "pad" in obj_type_densities else 0)
                data[f"segment{i}-density-orb"].append(obj_type_densities["orb"] if "orb" in obj_type_densities else 0)

    return pd.DataFrame(data)


def Predict_RandomForest(x: pd.DataFrame, y: pd.Series, apply_pca = False, balanced = False):
    if apply_pca:
        scalar = StandardScaler()
        x = scalar.fit_transform(x)
        pca = PCA(n_components='mle')
        x = pca.fit_transform(x)

        print(f"{pca.n_components_} components remaining.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    classifier = RandomForestClassifier(random_state=0)
    if balanced:
        classifier.class_weight = "balanced"
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")

    y_pred_train = grid_search.predict(x_train)
    print(confusion_matrix(y_train, y_pred_train, labels=["Easy", "Normal", "Hard", "Harder", "Insane", "Demon"]))
    print(f"Train Set Accuracy: {accuracy_score(y_train, y_pred_train)} ({(y_train == y_pred_train).sum()}/{x_train.shape[0]})\n")

    y_pred = grid_search.predict(x_test)
    print(confusion_matrix(y_test, y_pred, labels=["Easy", "Normal", "Hard", "Harder", "Insane", "Demon"]))
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred)} ({(y_test == y_pred).sum()}/{x_test.shape[0]})")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=["Easy", "Normal", "Hard", "Harder", "Insane", "Demon"]).plot()

    plt.show()


def Predict_NaiveBayes(x: pd.DataFrame, y: pd.Series):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    classifier = GaussianNB()

    scores = cross_val_score(classifier, x_train, y_train, cv=5)
    print(scores)
    print(scores.mean())

    classifier.fit(x_train, y_train)

    y_pred_train = classifier.predict(x_train)
    print(confusion_matrix(y_train, y_pred_train, labels=["Easy", "Normal", "Hard", "Harder", "Insane", "Demon"]))
    print(f"Train Set Accuracy: {accuracy_score(y_train, y_pred_train)} ({(y_train == y_pred_train).sum()}/{x_train.shape[0]})\n")

    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred, labels=["Easy", "Normal", "Hard", "Harder", "Insane", "Demon"]))
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred)} ({(y_test == y_pred).sum()}/{x_test.shape[0]})")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=["Easy", "Normal", "Hard", "Harder", "Insane", "Demon"]).plot()

    plt.show()


with open("hw1_dataset_raw.pkl", 'wb') as f:
    pickle.dump(GetDataset(), f)

with open("hw1_dataset_raw.pkl", 'rb') as f:
    raw_dataset: dict[int, list[dict]] = pickle.load(f)

print("Dataset Size:\n")
for diff in raw_dataset:
    print(f"{Difficulty(diff).name}: {len(raw_dataset[diff])}")
print("+ -------------")
print(f"Total: {sum([len(raw_dataset[val]) for val in raw_dataset])}")
df = ExtractFeatures(raw_dataset)
with open("hw1_dataset.pkl", 'wb') as f:
    pickle.dump(df, f)

with open("hw1_dataset.pkl", 'rb') as f:
    ds: pd.DataFrame = pickle.load(f)

print(ds["difficulty"].value_counts())
print(ds)
x = ds.iloc[:, 3:]
y = ds["difficulty"]
Predict_RandomForest(x, y, balanced=True)