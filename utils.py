import pandas as pd

TRAIN_COUNT = 500
TEST_COUNT = 1000
COLUMNS = [
    'id.orig_port',
    'id.resp_pport',
    'orig_bytes_count',
    'resp_bytes_count',
    'missed_bytes_count',
    'orig_pkts_count',
    'orig_ip_bytes_count',
    'resp_pkts_count',
    'resp_bytes'
]


def create_small(name, n, random_state=42):
    from zipfile import ZipFile

    zipped_data = ZipFile("data.zip", mode="r")

    good = pd.read_csv(zipped_data.open("Only_Benign.csv"))
    bad = pd.read_csv(zipped_data.open("Only_DDOS.csv"))

    assert good.shape[1] == bad.shape[1]

    print(good.columns)
    print(f"good count {good.shape[0]}, bad count {bad.shape[0]}")

    small_good = good.sample(n=n, random_state=random_state)
    small_bad = bad.sample(n=n, random_state=random_state)

    # just replace "-" with zeros
    small_good.replace("-", 0, inplace=True)
    small_bad.replace("-", 0, inplace=True)

    pd.DataFrame(small_bad).to_csv(f"bad_{name}.txt")
    pd.DataFrame(small_good).to_csv(f"good_{name}.txt")


def create_csv():
    create_small("train", TRAIN_COUNT)
    create_small("test", TEST_COUNT)


def get_train_dataset():
    good = pd.read_csv("good_train.txt")
    bad = pd.read_csv("bad_train.txt")

    df = pd.concat([good, bad])

    df_train = df[COLUMNS]
    y = df['Category'] == "Malicious"

    assert sum(y) == TRAIN_COUNT

    print("Good\n", good[COLUMNS].describe())
    print("\n\n")
    print("Bad\n", bad[COLUMNS].describe())

    return df_train, y


def get_test_dataset():
    good = pd.read_csv("good_test.txt")
    bad = pd.read_csv("bad_test.txt")

    df = pd.concat([good, bad])

    df_test = df[COLUMNS]
    y = df['Category'] == "Malicious"

    assert sum(y) == TEST_COUNT

    return df_test, y
