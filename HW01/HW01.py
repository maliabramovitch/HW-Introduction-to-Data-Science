import pandas as pd

""" 1. Contest data """
# A
data_csv = pd.read_csv("contestants.csv", header=0, index_col=['to_country_id'])
comp_num = data_csv.groupby('to_country_id')['place_final'].count()
place_median = data_csv.groupby('to_country_id')['place_final'].median()
num_first = data_csv[data_csv['place_final'] == 1].groupby('to_country_id')['place_final'].count()
prec = num_first / comp_num * 100
final_data = pd.concat([comp_num, place_median, num_first, prec], axis='columns',
                       keys=['compete_num', 'plac_final_med', 'first_place_num', 'first_place_for_compete'])
final_data.fillna(value=0, inplace=True)

# b
artists = data_csv.groupby('performer').count()
print(f"The performer that compete more than 3 times is: {'  -'.join(artists[artists['year'] > 3].index.tolist())}")

# c
composers = data_csv.groupby('composers').count()
print(
    f"The composers that composed more than 3 songs are: {'  -'.join(composers[composers['year'] > 3].index.tolist())}")

# d
israel_placing = data_csv[data_csv['to_country'] == 'Israel']['place_final'].value_counts().sort_values(ascending=False)
print(f"Israel placing: \n{israel_placing}")

""" Creating dictionary to convert Country's name from short to long """

