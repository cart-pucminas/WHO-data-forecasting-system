
from repositories import DataRepository, IndicatorRepository
from processing import InformationGain, NeuralNetwork, Outliers, MissingDatas, RandomForest
from utils import Utils

import pandas as pd
import time

from config.ConfigPrediction import config


def main():

    start = time.time()
    codePredictionIndicator = config.codePredictionIndicator
    
    datasPredictionDF = DataRepository.DataRepository().findDataIdYearCountryNameRegionNameDataValueByIndicatorNameOrIndicatorCode(indicatorCode=codePredictionIndicator)    
    datasToSendToPrediction = DataRepository.DataRepository().findDataIdYearCountryNameRegionNameDataValueByIndicatorNameOrIndicatorCode(indicatorCode=codePredictionIndicator)   
    
    if config.predictionIndicatorDimensions:
        for dimension in config.predictionIndicatorDimensions:
            dataAnalyzingByDimension = DataRepository.DataRepository().findDataIdDimensionName(codePredictionIndicator, dimension[0])
            datasPredictionDF = pd.merge(datasPredictionDF, dataAnalyzingByDimension, how='left', on=['dataId'])
            datasPredictionDF = datasPredictionDF[datasPredictionDF[dimension[0]] == dimension[1]]
            datasPredictionDF.rename(columns={codePredictionIndicator: codePredictionIndicator + "-" + str(dimension[1])}, inplace = True)
            datasToSendToPrediction = pd.merge(datasToSendToPrediction, dataAnalyzingByDimension, how='left', on=['dataId'])
            datasToSendToPrediction = datasToSendToPrediction[datasToSendToPrediction[dimension[0]] == dimension[1]]
            datasToSendToPrediction.rename(columns={codePredictionIndicator: codePredictionIndicator + "-" + str(dimension[1])}, inplace = True)
            codePredictionIndicator = codePredictionIndicator + "-" + str(dimension[1])
        
    datasPredictionDF = datasPredictionDF.drop(['dataId'], axis=1)
    datasToSendToPrediction = datasToSendToPrediction.drop(['dataId'], axis=1) 
    datasPredictionDF = Utils.joinDimensionsDatas(datasPredictionDF, codePredictionIndicator)
    datasToSendToPrediction = Utils.joinDimensionsDatas(datasToSendToPrediction, codePredictionIndicator)    
    
    indicatorsUsed = []
    indicatorsNotUsed = []
    
    if config.indicatorsAnalyzingCode:
        indicatorsAnalyzingCode = config.indicatorsAnalyzingCode
    else:
        indicators, indicatorsDf = IndicatorRepository.IndicatorRepository().findAllIndicatorsNotCodeAndNotPOP(config.codePredictionIndicator)
        indicatorsAnalyzingCode = indicatorsDf['code'].tolist()

    total = len(indicatorsAnalyzingCode)
    actual = 1
    indicatorsUsed = []
    indicatorsNotUsed = []
    for indicatorAnalyzingCode in indicatorsAnalyzingCode:
        missingDatasPercent = MissingDatas.missingDatasPercent(indicatorAnalyzingCode)
        if (missingDatasPercent == 0):
            indicatorsNotUsed.append([indicatorAnalyzingCode, 0, ''])
            # print("Not Including atribute -> " + str(indicatorAnalyzingCode) + '-> missingDatasPercent: ' + str(missingDatasPercent))
        else:
            dataAnalyzing = DataRepository.DataRepository().findDataIdYearCountryNameRegionNameDataValueByIndicatorNameOrIndicatorCode(indicatorCode=indicatorAnalyzingCode)
            dimensionsUsed = []
            dataAnalyzingToDimension = DataRepository.DataRepository().findDataIdYearCountryNameRegionNameDataValueByIndicatorNameOrIndicatorCode(indicatorCode=indicatorAnalyzingCode)
            for dimension in config.dimensionsPrediction:
                dataAnalyzingByDimension = DataRepository.DataRepository().findDataIdDimensionName(indicatorAnalyzingCode, dimension)
                if not dataAnalyzingByDimension.empty:
                    dataAnalyzingToDimension = pd.merge(dataAnalyzingToDimension, dataAnalyzingByDimension, how='left', on=['dataId'])
                    dimensionsUsed.append(dimension)

            def recursividade(base, pos, name, datas): 
                #stop condition
                if pos == len(dimensionsUsed):
                    base = base.drop(['dataId'], axis=1)
                    base = Utils.joinDimensionsDatas(base, name)
                    ig, iv, igr = InformationGain.informationGain(base, name, datasPredictionDF, codePredictionIndicator)
                    if ((ig > config.minInformationGain) and (missingDatasPercent < config.maxPercentMissingDatasToPrediction)):
                        if not base.empty:
                            indicatorsUsed.append([name, missingDatasPercent, ig])
                            print("Including atribute -> " + str(name) + '-> ig: ' + str(ig) + ' | iv: ' + str(iv) + ' | igr: ' + str(igr))
                            datas = pd.merge(datas, base, how='left', on=['regionName', 'year', 'countryName'])
                        else:
                            indicatorsNotUsed.append([name, missingDatasPercent, ig])
                        return datas
                    else:
                        # print("Not Including atribute -> " + str(name) + '-> ig: ' + str(ig) + ' | iv: ' + str(iv) + ' | igr: ' + str(igr))
                        indicatorsNotUsed.append([name, missingDatasPercent, ig])
                        return datas

                else:
                    newPos = pos+1
                    for x in base[dimensionsUsed[pos]].unique():
                        base_byUnique = base[base[dimensionsUsed[pos]] == x]
                        base_byUnique = base_byUnique.drop([dimensionsUsed[pos]], axis=1)
                        base_byUnique.rename(columns={name: str(name) + "-" + str(x)}, inplace = True)
                        datas = recursividade(base_byUnique, newPos, str(name) + "-" + str(x) , datas)
                    return datas
            
            if dimensionsUsed:
                datasToSendToPrediction = recursividade(dataAnalyzingToDimension, 0, indicatorAnalyzingCode, datasToSendToPrediction)
                
            dataAnalyzing = dataAnalyzing.drop(['dataId'], axis=1)
            dataAnalyzing = Utils.joinDimensionsDatas(dataAnalyzing, indicatorAnalyzingCode)
            ig, iv, igr = InformationGain.informationGain(dataAnalyzing, indicatorAnalyzingCode, datasPredictionDF, codePredictionIndicator)
            if ((ig > config.minInformationGain) and (missingDatasPercent < config.maxPercentMissingDatasToPrediction)):
                indicatorsUsed.append([indicatorAnalyzingCode, missingDatasPercent, ig])
                print("Including atribute -> " + str(indicatorAnalyzingCode) + '-> ig: ' + str(ig) + ' | iv: ' + str(iv) + ' | igr: ' + str(igr))
                datasToSendToPrediction = pd.merge(datasToSendToPrediction, dataAnalyzing, how='left', on=['regionName', 'year', 'countryName'])
            else:
                indicatorsNotUsed.append([indicatorAnalyzingCode, missingDatasPercent, ig])
                # print("Not Including atribute -> " + str(indicatorAnalyzingCode) + '-> ig: ' + str(ig) + ' | iv: ' + str(iv) + ' | igr: ' + str(igr))
        print(str(round((actual/total)*100, 2)) + "% Processing completed...")
        actual = actual+1


        
    indicatorsUsed = pd.DataFrame(indicatorsUsed,  columns=['indicatorAnalyzingCode', 'missingDatasPercent', 'ig'])
    indicatorsNotUsed = pd.DataFrame(indicatorsNotUsed,  columns=['indicatorAnalyzingCode', 'missingDatasPercent', 'ig'])
    indicatorsNotUsed.to_csv('../results/' + codePredictionIndicator.replace('/', '') + '_indicatorsNotUsed.csv' , sep=';',  encoding='utf-8')
    indicatorsUsed.sort_values(by=['ig'], inplace=True)
    indicatorsUsed.to_csv('../results/' + codePredictionIndicator.replace('/', '') + '_indicatorsUsedCompleted.csv' , sep=';',  encoding='utf-8')
    indicatorsUsed = indicatorsUsed.head(config.totalIndicatorsUseToPrediction)    
    indicatorsUsed.to_csv('../results/' + codePredictionIndicator.replace('/', '') + '_indicatorsUsed.csv' , sep=';',  encoding='utf-8')
    
    indicatorsUsed.drop_duplicates(subset=['indicatorAnalyzingCode'], inplace=True)

    datasToSendToPrediction.to_csv('../results/' + codePredictionIndicator.replace('/', '') + '_databaseAllComplete.csv', sep=';')

    datasToSendToPrediction = datasToSendToPrediction[['year', 'countryName', 'regionName', codePredictionIndicator] + indicatorsUsed['indicatorAnalyzingCode'].to_list()]

    datasPopulation = DataRepository.DataRepository().findDataIdYearCountryNameRegionNameDataValueByIndicatorNameOrIndicatorCode(indicatorName='Population', indicatorCode='POP') 
    # datasPopulation.rename(columns={'Population': 'pop'}, inplace = True)

    datasPopulation = datasPopulation.drop(['dataId'], axis=1)
    datasToSendToPrediction = pd.merge(datasToSendToPrediction, datasPopulation, how='left', on=['countryName', 'regionName', 'year'])
    
    datasToSendToPrediction = datasToSendToPrediction[datasToSendToPrediction['POP'].notna()]
    datasToSendToPrediction[codePredictionIndicator] = datasToSendToPrediction[codePredictionIndicator].astype(float)
    datasToSendToPrediction['POP'] = datasToSendToPrediction['POP'].astype(float)
    datasToSendToPrediction['percent'] = (datasToSendToPrediction[codePredictionIndicator]/(datasToSendToPrediction['POP'])*100)
    
    datasToSendToPrediction, outliers = Outliers.outliers(datasToSendToPrediction, config.outlierQparameter[0], config.outlierQparameter[1], 'percent')
    outliers.to_csv('../results/' + codePredictionIndicator.replace('/', '') + '_outliers.csv' , sep=';',  encoding='utf-8')
    resultDatabase = Utils.createGroups(datasToSendToPrediction)
    resultDatabase = resultDatabase.drop(['percent'], axis=1)
    resultDatabase = resultDatabase.drop([codePredictionIndicator], axis=1)
    resultDatabase = resultDatabase.drop(['POP'], axis=1)
    resultDatabase.rename(columns={'range': codePredictionIndicator}, inplace = True)
    resultDatabase.to_csv('../results/' + codePredictionIndicator.replace('/', '') + '_database.csv', sep=';')


    resultDatabase, indicatorsDatasMissingCompleted = MissingDatas.missingDatasComplete(resultDatabase)

    indicatorsDatasMissingCompleted = pd.DataFrame(indicatorsDatasMissingCompleted, columns=['indicatorAnalyzingName', 'missingDatasPercent'])
    indicatorsDatasMissingCompleted.to_csv('../results/' + codePredictionIndicator.replace('/', '') + '_missingCompleted.csv' , sep=';',  encoding='utf-8')
    
    resultDatabase.to_csv('../results/' + codePredictionIndicator.replace('/', '') + '_databaseFinal.csv', sep=';')

    for i in range(1, 6, 1):
        result, accuracy = NeuralNetwork.main(resultDatabase, codePredictionIndicator)
        confusionMatrix  = pd.crosstab(result['Actual Value'], result['Prediction'], rownames=['Actual Value/Prediction'], colnames=['Prediction'])
        result.to_csv('../results/' + codePredictionIndicator.replace('/', '') + '_result.csv' , sep=';',  encoding='utf-8')

        print("==============================================================================================================================================================")
        print("-> The Confusion Matrix Result: ")
        print(confusionMatrix)
        confusionMatrix.to_csv('../results/NeuralNetwork' + codePredictionIndicator + '_confusionMatrix_' + str(i) + '.csv' , sep=';',  encoding='utf-8')
        print("==============================================================================================================================================================")
        print("Accuracy: " + str(accuracy))
        print("==============================================================================================================================================================")
        print("==============================================================================================================================================================")
        print("==============================================================================================================================================================")

    for i in range(1, 6, 1):
        result, accuracy = RandomForest.main(resultDatabase, codePredictionIndicator)
        confusionMatrix  = pd.crosstab(result['Actual Value'], result['Prediction'], rownames=['Actual Value/Prediction'], colnames=['Prediction'])
        result.to_csv('../results/' + codePredictionIndicator.replace('/', '') + '_resultRF' + str(i) + '.csv' , sep=';',  encoding='utf-8')
        print("==============================================================================================================================================================")
        print("TEST " + str(i) + " :")
        print("-> The Confusion Matrix Result: ")
        print(confusionMatrix)
        confusionMatrix.to_csv('../results/RandomForest_' + codePredictionIndicator + '_confusionMatrix_' + str(i) + '.csv' , sep=';',  encoding='utf-8')
        print("==============================================================================================================================================================")
        print("Accuracy: " + str(accuracy))
        print("==============================================================================================================================================================")
        print("==============================================================================================================================================================")
        print("==============================================================================================================================================================")

    print("-> The indicators used to prediction were saved in the folder: results, file name: " + codePredictionIndicator + "_indicatorsUsed.csv")
    print("-> The indicators didn't use to prediction were saved in the folder: results, file name: " + codePredictionIndicator + "_indicatorsNotUsed.csv")
    print("-> The database outliers were saved in the folder: results, file name: " + codePredictionIndicator + "_outliers.csv")
    print("-> The database used to prediction was saved in the folder: results, file name: " + codePredictionIndicator + "_database.csv")
    print("-> The complete result was saved in the folder: results, file name: " + codePredictionIndicator + "_result.csv")
    print("-> The Confusion Matrix result was saved in the folder: results, file name: " + codePredictionIndicator + "_confusionMatrix.csv")

    print("==============================================")
    print("Total time: " + str(time.time()-start) + " sec.")


    