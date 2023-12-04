import numpy as np
import pandas as pd

def PrintTop10Players(df):
	df_1518 = df[df["p_year"] >= 2015]
	df_1518 = df_1518[df["p_year"]<=2018]
	print(df_1518.sort_values(by="H",ascending=False).head(10))
	print(df_1518.sort_values(by="avg",ascending=False).head(10))
	print(df_1518.sort_values(by="HR",ascending=False).head(10))
	print(df_1518.sort_values(by="OBP",ascending=False).head(10))

def HighestWARByPosition(df):
	df_2018 = df[df["p_year"]==2018]

	position = ["포수", "1루수", "2루수", "3루수", "유격수", "좌익수", "중견수", "우익수"]

	for tp in position:
  	print(df_2018[df_2018["tp"]==tp].sort_values(by="war",ascending=False).head(1))

def WhichHighestCorrWithSalary(df):
	df_money = df[["salary"]]
	category = ["R", "H", "HR", "RBI", "SB", "war" , "avg", "OBP", "SLG","salary"]

	content = df[category].corr()[["salary"]].sort_values(by="salary",ascending=False)
	print(content.iloc[1].name)
	print(content.iloc[1].salary)