from pyswip import Prolog
import pandas as pd

path_dataset = "data/heart_preprocessed.csv"

def creazioneFeatureIngegnerizzate():
    prolog = Prolog()
    df = pd.read_csv(path_dataset)

    print("Definisco la Knowledgebase")

    for index, row in df.iterrows():
        prolog.assertz(f'age({index}, {row["Age"]})')
        prolog.assertz(f'sex({index}, {row["Sex"]})')
        prolog.assertz(f'restingBP({index}, {row["RestingBP"]})')
        prolog.assertz(f'cholesterol({index}, {row["Cholesterol"]})')
        prolog.assertz(f'maxHR({index}, {row["MaxHR"]})')

    prolog.assertz("isOverMaxRate(IdPaziente):- sex(IdPaziente, Sesso), age(IdPaziente, Eta), maxHR(IdPaziente, HR), Sesso =:= 1, 220 - Eta < HR")
    prolog.assertz("isOverMaxRate(IdPaziente):- sex(IdPaziente, Sesso), age(IdPaziente, Eta), maxHR(IdPaziente, HR), Sesso =:= 0, 226 - Eta < HR")
    prolog.assertz("isOverMaxCholesterol(IdPaziente):- age(IdPaziente, Eta), cholesterol(IdPaziente, CH), Eta >= 20, 200 > CH")
    prolog.assertz("isOverMaxCholesterol(IdPaziente):- age(IdPaziente, Eta), cholesterol(IdPaziente, CH), Eta =< 19, 170 > CH")
    prolog.assertz("isOverMinCholesterol(IdPaziente):- age(IdPaziente, Eta), cholesterol(IdPaziente, CH), Eta >= 20, 125 < CH")


    for index, row in df.iterrows():
        if bool(list(prolog.query(f"isOverMaxRate({index})"))):
            df.at[index, 'isOverMaxRate'] = "1"
        else:
            df.at[index, 'isOverMaxRate'] = "0"

        if bool(list(prolog.query(f"isOverMaxCholesterol({index})"))):
            df.at[index, 'isOverMaxCholesterol'] = "1"
        else:
            df.at[index, 'isOverMaxCholesterol'] = "0"

        if bool(list(prolog.query(f"isOverMinCholesterol({index})"))):
            df.at[index, 'isOverMinCholesterol'] = "1"
        else:
            df.at[index, 'isOverMinCholesterol'] = "0"


    print("Feature ingegnerizzate create con successo")

    df.to_csv("data/heart_preprocessed.csv")

    print("Dataset salvato con successo!")
