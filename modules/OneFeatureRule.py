# adaptado de https://github.com/mfreyeso/oner-scratch/blob/master/Scratch%20One%20Rule.ipynb por Josenalde Oliveira, 2025,april
import pandas as pd


class OneFeatureRule(object):

    def __init__(self):
        self.ideal_variable = None
        self.ideal_variable_index = None
        self.max_accuracy = 0

    def fit(self, X, y):
        response = list()
        result = dict()  # cria um dicionario vazio (sem chaves, sem valores)

        dfx = pd.DataFrame(X)

        for feature in dfx:  # i é a coluna (tamanho, quantidade)
            # print(feature) # o nome da coluna (feature)
            # print(dfx[feature]) #o conteudo da coluna (valores desta coluna)
            # result[str(i)] = dict()

            # cria um dataframe com duas colunas, sendo a feature atual + a coluna alvo à direita
            join_data = pd.DataFrame({"feature": dfx[feature], "target": y})
            # print(join_data)

            # cria uma tabela cruzada (produto cartesiano) para depois obter a frequência
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html
            cross_table = pd.crosstab(join_data.feature, join_data.target)
            # print(cross_table)
            # para cada classe da feature atual, obtém o índice (nome da coluna) com maior contagem IDXMAX
            summary = cross_table.idxmax(axis=1)

            # guarda as regras ganhadoras para esta feature
            result[feature] = dict(summary)
            # print(dict(summary))

            counts = 0
            # iterar no dataframe composto pela feature atual e pela coluna alvo (row é a observação/linha atual)
            # calcular a acurácia
            for idx, row in join_data.iterrows():
                # print(row)
                # aqui irá percorrer a coluna alvo, verificando se a classificação é igual à classe ganhadora para esta classe da feature
                # se for, adiciona um, significa que está coerente com a regra ganhadora
                if row['target'] == result[feature][row['feature']]:
                    counts += 1
                # print(str(row['target']) + '==' + str(result[feature][row['feature']]) + ' ' + str(counts))
            # acertos/total de linhas do df
            accuracy = (counts/len(y))

            # atualiza a maior acurácia, para indicar qual a melhor feature para as regras
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                self.ideal_variable = feature
                self.ideal_variable_index = dfx.columns.get_loc(feature)

            result_feature = {"feature": feature,
                              "accuracy": round(accuracy, 2), "rules": result[feature]}
            response.append(result_feature)

        return response

    # def predict(self, X=None):
    #    self_ideal_variable = self.ideal_variable + 1 #to-do

    def __repr__(self):
        if self.ideal_variable != None:
            txt = "Melhor variavel de decisão para seus dados: " + \
                str(self.ideal_variable)
        else:
            txt = "Ainda não encontrei melhor variável de decisão, tente executar o treino novamente"
        return txt
