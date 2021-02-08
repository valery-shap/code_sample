#!/usr/bin/env python
# coding: utf-8



# # Описание проекта
# 
# Подготовка прототипа модели машинного обучения для «Цифры». Компания разрабатывает решения для эффективной работы промышленных предприятий.
# 
# Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. В вашем распоряжении данные с параметрами добычи и очистки. 
# 
# Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.
# 
# Необходимо:
# 
# 1. Подготовить данные;
# 2. Провести исследовательский анализ данных;
# 3. Построить и обучить модель.

# # 1. Подготовка данных

import pandas as pd
pd.set_option('display.max_columns', 100)
data_tr = pd.read_csv('/datasets/gold_recovery_train.csv')
data_tr.info()
data_test = pd.read_csv('/datasets/gold_recovery_test.csv')
data_test.info()
data_full = pd.read_csv('/datasets/gold_recovery_full.csv')
data_full.info()


# всего 87 признаков, в тестовой всего 53. есть пропуски. много. в разных столбцах.

tr_columns = data_tr.columns.get_values()
tr_columns = tr_columns.tolist()
test_columns = data_test.columns.get_values()
test_columns = test_columns.tolist()

#  посмотрю, каких столбцов нет в тестовой

extra_column = []
for name in tr_columns:
    if name not in test_columns and name != 'final.output.recovery' and name != 'rougher.output.recovery':
        extra_column.append(name)
extra_column


#  почти все параметры относятся к типу output,но есть 4 из calculation. может они как раз нужны, для того чтобы  output  посчитать. может для того, чтобы не было утечки признаков.


# Проверьте, что эффективность обогащения рассчитана правильно. Вычислите её на обучающей выборке для признака rougher.output.recovery. Найдите MAE между вашими расчётами и значением признака. Опишите выводы.

# для этого стоит определить, как называются столбцы с долей золота до/после флотации и в отвальных хвостах.


# хочу посмотреть на маленькой выборке, как считать

data_tr_recov = data_tr[['rougher.input.feed_au', 'rougher.output.tail_au','rougher.output.concentrate_au', 'rougher.output.recovery']].head()
data_tr_recov['new_recovery'] = (data_tr_recov['rougher.output.concentrate_au']*(data_tr_recov['rougher.input.feed_au'] - data_tr_recov['rougher.output.tail_au'])) / (data_tr_recov['rougher.input.feed_au'] * (data_tr_recov['rougher.output.concentrate_au'] - data_tr_recov['rougher.output.tail_au'])) * 100
# совпадают
def mae(table_recovery, my_recovery):
    mae = (sum(abs(table_recovery - my_recovery)))/table_recovery.shape[0]
    return mae
data_tr_recov['mae'] = mae(data_tr_recov['rougher.output.recovery'], data_tr_recov['new_recovery'])
data_tr_recov['mae'].value_counts()


#  а должно было быть 0. везде  mae очень маленькое и одинаковое. теперь попробую на всей таблице

# скорей всего надо убрать пропуски, чтобы посчиталось

# строки без целевых признаков удалю. остальные заполню ближайшим значением

data_tr_check_sum = data_tr.drop(['rougher.output.recovery', 'final.output.recovery' ], axis = 1)
data_tr_check_sum = data_tr_check_sum.dropna()
data_tr_check_sum.info()

data_tr = data_tr.dropna(subset = ['rougher.output.recovery', 'final.output.recovery' ])
data_tr.info()


data_full = data_full.dropna(subset = ['rougher.output.recovery', 'final.output.recovery' ])
data_full.info()

data_tr.info()

data_tr_for_check = data_tr.dropna()
data_tr_for_check.info()

data_tr = data_tr.fillna(method = 'ffill')
data_tr.info()


# расчет MAE

def check_recovery(data):
    data['new_recovery'] = (data['rougher.output.concentrate_au']*(data['rougher.input.feed_au'] - data['rougher.output.tail_au'])) / (data['rougher.input.feed_au'] * (data['rougher.output.concentrate_au'] - data['rougher.output.tail_au'])) * 100
    data['mae'] = mae(data['rougher.output.recovery'], data['new_recovery'])
    return data


data_tr_for_check = check_recovery(data_tr_for_check)
data_tr_for_check['mae'].value_counts()


# значит значения во всей таблице совпали.

# теперь надо разобраться с тестовой выборкой

data_test.info()

data_test = data_test.fillna(method = 'ffill')
data_test.info()


# нужно добавить целевые признаки в тестовую выборку, видимо по дате

data_full.columns


# # 2. Анализ данных

# Посмотрите, как меняется концентрация металлов (Au, Ag, Pb) на различных этапах очистки. Опишите выводы.

# rougher.input.feed_ag,rougher.output.concentrate_ag  , primary_cleaner.output.concentrate_ag,final.output.concentrate_ag


data_ag = data_full[['rougher.input.feed_ag','rougher.output.concentrate_ag'  , 'primary_cleaner.output.concentrate_ag', 'final.output.concentrate_ag']]
data_ag.head(10)



data_full['rougher.input.feed_ag'].median()
data_full['rougher.output.concentrate_ag'].median()
data_full['primary_cleaner.output.concentrate_ag'].median()
data_full['final.output.concentrate_ag'].median()


# одержание Ag снизилось к финалу. 
import seaborn as sns
ax = sns.boxplot(data= data_full[['rougher.input.feed_ag','rougher.output.concentrate_ag'  , 'primary_cleaner.output.concentrate_ag', 'final.output.concentrate_ag']] )
ax.set_xticklabels(ax.get_xticklabels(),rotation=15, ha='right')


ax = sns.boxplot(data= data_full[['rougher.input.feed_au' ,'rougher.output.concentrate_au'  , 'primary_cleaner.output.concentrate_au', 'final.output.concentrate_au']] )
ax.set_xticklabels(ax.get_xticklabels(),rotation=15, ha='right')


# концентрация золота растет



ax = sns.boxplot(data= data_full[['rougher.input.feed_pb','rougher.output.concentrate_pb'  , 'primary_cleaner.output.concentrate_pb', 'final.output.concentrate_pb']] )
ax.set_xticklabels(ax.get_xticklabels(),rotation=15, ha='right')


# концентрация Pb увеличилась к концу

# Судя по графикам золота становится все больше и больше с каждым этапом, что скорей всего один из самых главных показателей, что все идет хорошо.(остается только вопрос насколько больше). 
# Непонятно, почему после флотации Ag становится больше, может быть так и должно быть, а может быть что-то не так на производстве. надо уточнять. далее концентрация Ag падает, что хорошо, мы же золото добываем. Концентрация Pb возрастает с каждым этапом, выходит на плато после первой очистки. Скорей всего для того чтобы оценить правильность этого распределения нужно знать, а как должно быть, если все очищается по плану. Мне казалось, что все ,кроме золота, должно уменьшаться.


# Сравните распределения размеров гранул сырья на обучающей и тестовой выборках. Если распределения сильно отличаются друг от друга, оценка модели будет неправильной.

# интересно, а почему только это распределение надо смотреть

# размеры гранул есть на разных этапах. предполагаю, что имели ввиду rougher.input.feed_size


data_tr['rougher.input.feed_size'].describe()

sns.distplot(data_tr['rougher.input.feed_size'])
sns.distplot(data_test['rougher.input.feed_size'])
data_test['rougher.input.feed_size'].describe()


# медианы отличаются в пределах отклонений. вроде все норм.

data_full.info()

data_full_check = data_full.dropna()
data_full_check.info()

import matplotlib.pyplot as plt

def sum_elements(data):
    data['rougher_input_sum'] = data['rougher.input.feed_ag'] + data['rougher.input.feed_au'] + data['rougher.input.feed_pb'] + data['rougher.input.feed_sol']
    data['rougher_output_sum'] = data['rougher.output.concentrate_ag'] + data['rougher.output.concentrate_au'] + data['rougher.output.concentrate_pb'] +data['rougher.output.concentrate_sol']  
    data['primary_output_sum'] = data['primary_cleaner.output.concentrate_ag'] + data['primary_cleaner.output.concentrate_au'] + data['primary_cleaner.output.concentrate_pb'] + data['primary_cleaner.output.concentrate_sol']
    data['final_otput_sum'] = data['final.output.concentrate_ag'] + data['final.output.concentrate_au'] + data['final.output.concentrate_pb'] + data['final.output.concentrate_sol']
    sns.distplot(data['rougher_input_sum'], label = 'rougher_input')
    sns.distplot(data['rougher_output_sum'], label = 'rougher_output_sum')
    sns.distplot(data['primary_output_sum'], label = 'primary_output_sum')
    sns.distplot(data['final_otput_sum'], label = 'final_output')
    plt.legend()

sum_elements(data_full_check)


# выбросов не вижу.

sum_elements(data_tr_check_sum)


# data_tr_check_sum это таблица , из которой сначала удалила все столбцы с целевыми признаками, а потом убрала строки со всеми пропусками в нецелевых свойствах. Выбросов тоже мин количество.


good_ids = data_full['date']
good_ids = good_ids.tolist()
good_ids


# надо оставить в обучающей только те ids, которые остались после удаления выбросов

data_test = data_test.query('date in @good_ids')
data_test.shape
data_tr = data_tr.query('date in @good_ids')
data_tr.shape


# теперь стоит добавить целевые признаки к тестовой выборке:rougher.output.recovery, final.output.recovery

data_test = data_test.merge(data_full[['date','rougher.output.recovery','final.output.recovery']], left_on = 'date', right_on = 'date')
data_test.head()
data_full[data_full['date'] == '2016-09-01 04:59:59']


# 1. удалила строки с отсутвующим значением целевых признаков
# 2. пропуски в остальных столбцах заменила на ближайшие
# 3. удалила выбросы на основе анализа распределений суммарных концентраций всех веществ
# 4. добавила целевые признаки в тестовую выборку

# # 3. Модель

# Напишите функцию для вычисления итоговой sMAPE

# я не провела подготовку данных к построению модели. Надо скорей всего все численные признаки привести к одному масштабу. и в тренировочной и в обучающей. rougher.output.recovery , final.output.recovery.
# у меня не получилось, так как количество features в train и test разное. удалю из train.

data_tr = data_tr.drop(extra_column,axis = 1)
data_tr.shape
data_test.shape
from sklearn.preprocessing import StandardScaler
features_train = data_tr.drop(['date','rougher.output.recovery','final.output.recovery'] , axis=1)
targets_train = data_tr[['rougher.output.recovery','final.output.recovery']]
features_test = data_test.drop(['date','rougher.output.recovery','final.output.recovery'] , axis=1)
targets_test = data_test[['rougher.output.recovery','final.output.recovery']]
scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)


# функция метрики должна быть все-таки одна и без гридсеч у меня получилось ее реализовать
def metric(answers, predictions):
    #sMAPE = (sum((abs(answers - predictions)/((abs(answers) + abs(predictions))/2))))/answers.shape[0]* 100
    #sMAPE = abs(answers + predictions) / answers.shape[0]
    answers = pd.DataFrame(answers)
    predictions = pd.DataFrame(predictions, index=answers.index)
    #predictions = pd.DataFrame(predictions)
    first = abs(answers['rougher.output.recovery'] - predictions[0])
    second = (abs(answers['rougher.output.recovery']) + abs(predictions[0]))/2
    third = sum(first/second)
    forth = third / answers.shape[0] * 100
    first_2 = abs(answers['final.output.recovery'] - predictions[1])
    second_2 = (abs(answers['final.output.recovery']) + abs(predictions[1]))/2
    third_2 = sum(first_2/second_2)
    forth_2 = third_2 / answers.shape[0] * 100
    final_metric = 0.25*forth + 0.75*forth_2
    return final_metric
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state = 12345)
model.fit(features_train, targets_train)
predicted_test = model.predict(features_test)
#predicted_test = pd.DataFrame(predicted_test)
final = metric(targets_test, predicted_test)
final


# теперь попробую с GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
param_grid = { 
    'n_estimators': [5, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8]
}
scorer = make_scorer(metric,greater_is_better = False)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5,scoring = scorer)
grid_search.fit(features_train, targets_train)
grid_search.best_params_, grid_search.best_score_


# лучшие параметры другие 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 100}
# и лучшая метрика -10.102. это какой-то скор, не связанный с sMAPE?


model = RandomForestRegressor(max_depth = 6, n_estimators = 100, max_features = 'log2', random_state = 12345)
model.fit(features_train, targets_train)
predicted_test = model.predict(features_test)
final = metric(targets_test, predicted_test)
final


# на тестовой стало 9.41370626003541

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 12345)
model.get_params().keys()

param_grid = { 
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [5,100]
}
scorer = make_scorer(metric, greater_is_better = False)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5,scoring = scorer)
grid_search.fit(features_train, targets_train)
grid_search.best_params_

model = DecisionTreeRegressor(max_depth = 5, max_features = 'log2', random_state = 12345)
model.fit(features_train, targets_train)
predicted_test = model.predict(features_test)
final = metric(targets_test, predicted_test)
final

from sklearn.dummy import DummyRegressor
model = DummyRegressor()
model.fit(features_train, targets_train)
predicted_test = model.predict(features_test)
final = metric(targets_test, predicted_test)
final


# # Выводы

# лучший sMAPE у леса на тестовой 9.41370626003541. у дерева на тестовой 9.539779143797084.  у dummyregressor 10.289630376021057.
# если чем меньше показатель лучше, значит лес показал наилучшие значения.
