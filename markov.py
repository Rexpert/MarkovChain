import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)

DEBUG = False
SEED = 1234
RANDOM_LEVEL = 0.5


def classify(str):
    if "/harga/" in str:
        type = "PDP"
    elif "/search/" in str:
        type = "Search Page"
    elif "/checkout/" in str:
        type = "Checkout Page"
    else:
        type = "PLP"
    return type


def clean(data, str):
    if DEBUG:
        data = data.sample(10, random_state=SEED)
    else:
        data = randomize(data)
    col_name = str.title() + " Page"
    data = data.assign(page_type=lambda df: df[col_name].map(
        lambda pg: classify(pg))).iloc[:, [2, 1]]
    data["Pageviews"] = data.groupby(["page_type"])[
        "Pageviews"].transform("sum")
    return data.drop_duplicates(subset=["page_type"])


def randomize(df):
    df_sample = df.sample(frac=RANDOM_LEVEL, random_state=SEED)
    size = len(df_sample)
    np.random.seed(SEED)
    percent = np.random.uniform(-RANDOM_LEVEL, RANDOM_LEVEL, size)
    value = df_sample["Pageviews"] * (1 + percent)
    replace = np.absolute(np.around(value)).astype(int)
    df.loc[replace.index, "Pageviews"] = replace
    return df


# Read Raw Data
page = pd.read_excel("data/raw.xlsx", sheet_name="previousPage")
landing = pd.read_excel("data/raw.xlsx", sheet_name="landing")
exit = pd.read_excel("data/raw.xlsx", sheet_name="exit")

# Cleaning page dataframe
if DEBUG:
    page = page.sample(10, random_state=SEED)
else:
    page = randomize(page)
page = page[page["Previous Page Path"] != "(entrance)"]\
    .assign(
        page_type=lambda df: df["Page"].map(lambda pg: classify(pg))
).assign(
        previous_page_type=lambda df: df["Previous Page Path"].map(
            lambda pg: classify(pg))
).iloc[:, [4, 3, 2]]

page["Pageviews"] = page\
    .groupby(["page_type", "previous_page_type"])["Pageviews"]\
    .transform("sum")
page_type = page.drop_duplicates(subset=["page_type", "previous_page_type"])

# Cleaning Landing & Exit Dataframe
landing_type = clean(landing, "landing")\
    .assign(previous_page_type="Start")\
    .iloc[:, [2, 0, 1]]
exit_type = clean(exit, "exit")\
    .assign(previous_page_type=lambda df: df["page_type"])\
    .assign(page_type="End")\
    .iloc[:, [2, 0, 1]]


# Transition Matrix in Pandas
transition_matrix = page_type\
    .append(landing_type)\
    .append(exit_type)\
    .pivot(index="previous_page_type", columns="page_type", values="Pageviews")\
    .fillna(0)\
    .astype(int)

transition_matrix = transition_matrix\
    .append(pd.Series(0, index=transition_matrix.columns, name="End"))\
    .assign(Start=0)

transition_matrix.loc["End", "End"] = transition_matrix.to_numpy().sum()

order = ["Start", "Checkout Page", "PDP", "PLP", "Search Page", "End"]
transition_matrix = transition_matrix\
    .reindex(order, axis=0)\
    .reindex(order, axis=1)

if DEBUG:
    print(transition_matrix)
else:
    transition_matrix.to_csv("result/transition_matrix.csv")

# np_matrix = pd.read_csv("result/transition_matrix.csv").iloc[:, 1:].to_numpy()
np_matrix = transition_matrix.to_numpy()

row_sum = np.sum(np_matrix, axis=1)
# transition matrix, P
P = np_matrix / row_sum[:, None]

Q = P[:5, :5]
R = P[:5, 5]

I = np.identity(len(Q))
# Fundamental matrix, N = (I - Q) ^ -1
N = np.linalg.inv(I - Q)

# Expected Number of time, t = N * 1
t = np.sum(N, axis=1)
