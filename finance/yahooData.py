import fix_yahoo_finance as yf
import os
import utils.logger

def getStockData(stockName, start, end, dst):
    """
    :param stockName: The Stock Name
    :param start: The start date
    :param end: The end date
    :param dst: The dest folder for store
    :return:
    """
    data = yf.download(stockName,start, end)
    fileName = os.path.join(dst,stockName)+'.csv'
    data.to_csv(fileName)

if __name__ == "__main__":
    getStockData('IBM', '2001-01-01', '2018-04-08', '../data/')