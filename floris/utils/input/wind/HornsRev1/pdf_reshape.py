import numpy as np
import pandas as pd


def main():
    wind_pdf = pd.read_csv('./pdf_horns_1_2.csv', index_col=0)
    speeds, directions = wind_pdf.index.values, wind_pdf.columns.values
    print(wind_pdf)
    # print(wind_pdf.values[0, :20])

    wind_rose = np.zeros((len(speeds) * len(directions), 3))
    ws, wd = np.meshgrid(speeds, directions, indexing='ij')
    ws, wd = ws.flatten(), wd.flatten().astype(np.float64)
    freq_val = wind_pdf.values.reshape((len(speeds) * len(directions),))
    # print(freq_val[:20])
    wind_rose[:, 0], wind_rose[:, 1], wind_rose[:, 2] = ws, wd, freq_val
    wind_rose = pd.DataFrame(wind_rose, columns=['ws', 'wd', 'freq_val'])
    print(wind_rose)
    wind_rose.to_csv('./wind_rose_horns_1_2.csv', index=0)




if __name__ == "__main__":
    main()