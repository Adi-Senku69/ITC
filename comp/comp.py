import pandas as pd


def comp(threshold, img_path, Q=0):
    without_q = pd.read_csv("comp/Without_Q.csv")
    with_q = pd.read_csv("comp/With_Q.csv")
    print("Q: Default (8)")
    print(without_q)
    print("\n\nQ:4 Delta: 5\n\n", with_q.iloc[:, [0, 1]])
    print("\n\nQ:8 Delta: 21\n\n", with_q.iloc[:, [0, 2]])
    print("\n\nQ:16 Delta: 85\n\n", with_q.iloc[:, [0, 3]])

if __name__ == "__main__":
    comp(10, 2, 0)
