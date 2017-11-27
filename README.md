**kNN - классификатор ближайших соседей**
=======================================

*Классификатор ближайших соседей* –  простейший метрический классификатор, основанный на оценивании сходства объектов. Классифицируемый объект относится к тому классу, к которому принадлежат ближайшие к нему объекты обучающей выборки.

В sсikit-learn реализовано два классификатора ближайших соседей: *KNeighborsClassifier (k - целое, число соседей)* использует *k* ближайших соседей каждого объекта. *k* указывается пользователем и может быть только целым числом. *RadiusNeighborsClassifier (r - действительное число, радиус)* с помощью радиуса ближайших соседей определяет количество соседей в пределах фиксированного радиуса *r* для каждого объекта обучения в случае неравномерно распределенных данных. Значение r указывается пользователем и является нецелым значением.

В задаче классификации ближайших соседей используются однородные веса: значение веса для каждой точки равно количеству «голосов» ближайших соседей этой точки. Бывают случаи, когда лучше соседям, расположенным ближе всего к объекту, присваивать большее значение весов, чем более отдаленным точкам. Все это выполнимо с помощью использования переменной *weight* = «равномерность»: каждому соседу присваиваются одинаковые веса (все точки в каждой окрестности взвешены одинаково).

*Weight* = «расстояние»: значение веса равно обратному расстоянию от классифицируемой точки до всех остальных точек окрестности. Обычно, для вычисления весов используется функция расстояния.

В программной реализации метода в качестве входных данных использовалась выборка:
```
wine = datasets.load_wine()
```

В первом случае *weight = 'uniform'*, т.е. все точки имеют одинаковый вес.

![](https://raw.githubusercontent.com/Kursaitova/KNN/master/wine1.PNG "kNN")

Во втором случае *weight = 'distance'*, т.е. весовая функция представляет собой убывающую последовательность вещественных весов.

![](https://raw.githubusercontent.com/Kursaitova/KNN/master/wine2.PNG "kNN")

```
KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)
```
Код программы:

```
# количество соседей
n_neighbors = 15

# импортируем данные выборки 
iris = datasets.load_wine()

# будем рассматривать только первые две характеристики Вина 
X = iris.data[:, :2]
y = iris.target

h = .02  # Задаем размер шага

# Задаем цветовые характеристики карт 
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) #цвет областей
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) #цвет Ирисов из выборки 
```

**Параметры:**

**n_neighbors**: целое. Это количество соседей, используемых по умолчанию для запросов kneighbors.

```
n_neighbors = 15
```

**Weights**: строка или заданная пользователем функция. Возможные значения:

* *“uniform”*: равномерные веса. Все точки имеют одинаковый вес.

* *“distance”*: вес точек указывает на обратное расстояние. В этом случае, более близкие соседи точки будут иметь большее влияние, чем соседи, находящиеся дальше.

* *[callable]*: заданная пользователем функция, которая в качестве входных данных использует массив расстояний и возвращает массив такого же размера, содержащий веса.

```
for weights in ['uniform', 'distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
```

**algorithm** : *{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}*. Алгоритм, используемый для вычисления ближайших соседей:

* *“ball_tree”* использует класс BallTree.

* *“kd_tree”* использует класс KDTree.

* *“brute”* использует поиск перебором.

* *“auto”* пытается подобрать наиболее подходящий алгоритм, опираясь на значения, переданные методу *fit*.

**leaf_size**: целое. Это значение передается в BallTree или KDTree. Может повлиять на скорость построения запроса, а также на память, необходимую для хранения дерева.
	
**p**: целое. Параметр метрики Минковского. Для *р=1* применяется «манхэттонское расстояние», и эвклидово расстояние для *р=2*. Для произвольного p используется расстояние Минковского.
	
**metric**: строка или пользовательская функцияю Это расстояние, используемое в дереве. По умолчанию используется метрика Минковского, в для *р=2* – Эвклидова метрика.
	
**metric_params**: Дополнительные аргументы для метрики.

**n_jobs**: целое. Число параллельных потоков для поиска соседей. Если равно *-1*, то число потоков = числу ядер процессора.

**Методы:**

* *fit(X, y)*: подгонка модели, используя *Х* в качестве обучающей выборки, и *у* в качестве целевых значений.

* *get_params([deep])*: получает данные для этого метода оценивания.

* *kneighbors([X, n_neighbors, return_distance])*:  находит *k*-соседей точки.

* *kneighbors_graph([X, n_neighbors, mode])*: вычисляет (взвешенный) граф *k*-соседей для точек из *Х*.

* *predict(X)*: определяет метки классов для предоставленных данных.

* *predict_proba(X)*: возвращаются оценки вероятности для проверки данных *Х*.

* *score(X, y[, sample_weight])*: возвращает среднюю точность тестовых данных и меток.

* *set_params(params)*: необходимо задать параметры для этой оценки.

