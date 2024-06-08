import pandas as pd

""" 1. Contest data """
# A
data_csv = pd.read_csv("contestants.csv", header=0)
data_csv.set_index('to_country_id', drop=False)
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


""" 2. Creating dictionary to convert Country's name from short to long """
codeToCountry = data_csv[['to_country_id', 'to_country']]
codeToCountry = codeToCountry.drop_duplicates(keep='first', ignore_index=True)
codes = codeToCountry['to_country_id'].to_frame()
names = codeToCountry['to_country'].to_frame()

code_to_country = {}
for key, value in zip(codes.values, names.values):
    code_to_country[key[0]] = value[0]

