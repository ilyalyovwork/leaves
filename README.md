# Tree leaves segmentation

## Содержание

1. [Описание проекта](#описание-проекта)
2. [Описание метода](#описание-метода)
3. [Подготовка данных](#подготовка-данных)
4. [Структура проекта](#структура-проекта)
5. [Результаты](#результаты)

## Описание проекта

В данной работе решается задача сегментации (обнаружения) опавших листьев в осенний период. Разработанный метод может быть применен в качестве бейзлайна для установки на роботов, предназначенных подметать листья на улице (дворников). Хоть пока таких и не существует:)

Данная работа не несет научный характер и создана для ознакомления с задачей компьютерного зрения - сегментацией изображений и изучения библиотек [pytorch](https://pytorch.org/docs/stable/index.html) и albumentations(https://albumentations.ai/docs/).

## Описание метода

В данной работе в качестве модели используется [U-Net](https://arxiv.org/abs/1505.04597) c предобученным энкодером [VGG11](https://arxiv.org/abs/1409.1556v6) на датасете [Imagenet](https://image-net.org/). VGG11 взята из библиотеки [pytorch](https://pytorch.org/vision/stable/models.html).

Такой архитектура сети называется [Ternausnet](https://arxiv.org/abs/1801.05746).

Поскольку проект делается в домашних условиях, то из-за отсутствии мощной видеокарты, для обучения модели используется видеокарта из google collab, которая выдается на ограниченный промежуток времени  с небольшим объемом оперативной памяти (около 11 gb). В следствие чего в проекте используется U-Net11, а не более продвинутые тяжелые модели, такие как U-Net16, Feature Pyramid Networks ([FPN](https://arxiv.org/pdf/1612.03144.pdf)) и другие.

## Подготовка данных

Для обучения сети было вручную размечено 36 фотографий опавших листьев, лежащих на тратуаре, асфальте, траве, детской площадке. Фотографии были сделаны под разным освещением (в разные промежутки дневного времени) с разрешением 3024x4032 пикселя.

Каждое изображение было поделено на 12 частей - по 1008x1008 пикселей на окно и сжато (resize) до 320x320. Сделано это было опять же из-за недостатка оперативной памяти в видеокарте (большие картинки просто не помещались в память). Размер 320x320 был подобран так, чтобы максимально использовать оперативную память, чтобы увеличить размер батча, при этом соблюдая ограничение сети: размеры картинки должны быть кратны 2^n=32, где n - количество пулинг слоев в сети (в текущей 5).

Также, отличительной особенностью данной работы является то, что перед обучением на основном датасете, полученным самостоятельно, модель предобучалась на датасете [leafsnap](http://leafsnap.com/dataset/), представляющее собой разметку 185 видов листьев деревьев на белом фоне (23147 изображений по 600x800 пикселей). Сделано это было, чтобы модель на ранних этапах смогла выявить закономерность в структуре листьев, чтобы лучше в дальнейшем предсказывать результаты.

## Структура проекта

```
.
├── dataset.py // Формирование датасета из каталога картинок
├── draw.py // Визуализация изображения с наложенной маской
├── ternausnet 
|    ├── __init__.py
|    └── ternausnet.py // Модель сети
├── img_transformations.py // Преобразования над картинками (albumentations, сшивание/расшивание)
├── loss.py // Имплементация функии потерь
├── metrics.py // Метрики dice и jaccard
├── train_val_paths.py // Получение всех путей к файлам и разделение их на train/val
├── train.py // Обучение модели
├── My_leaves dataset observe.ipynb // Обзор street dataset и аугментаций для него
├── Research.ipynb // Проведение исследования
├── Demo.ipynb // Результаты работы (%TODO)
├── README.md
└── validation.py // Валидация модели
```

## Результаты

| | Init model | Model trained on leafsnap  | Model trained on street leaves |
| :---- |:----|:----|:----|
| Train loss  | - | 0.034 | 0.061 |
| Val loss on leafsnap | 0.946 | 0.049 | - |
| Jaccard on leafsnap  | 0.107 | 0.879 | - |
| Val loss on street dataset | 1.293 | 0.301 | 0.045 |
| Jaccard on street dataset  | 0.040 | 0.378 | 0.809 |

Начальная U-Net11 с предобученным энкодером показала следующие результаты:
- leafsnap Dataset. Val_loss=0.946, Jaccard=0.107.
- street leaves Dataset. Valid loss: 1.293, Jaccard: 0.040.

Пример сегментации листа:
![image](https://user-images.githubusercontent.com/85474856/136532282-e05eb464-97d5-4dc7-bc24-2154e478b80c.png)

После чего была обучена модель на датасете leafsnap. При этом ошибка на валидационной выборке равна 0.049. А Jaccard метрика - 0.879.

Пример сегментации листа из датасета leafsnap:
![image](https://user-images.githubusercontent.com/85474856/136532692-8797f8c5-aa92-4b58-b179-dd895d0f1510.png)


Также, была произведена оценка качества модели на собственном датасете (street dataset).
В результате чего ошибка на валидации составила 0.301. А Jaccard метрика - 0.378, что гораздо лучше результатов изначальной модели. Из чего можно сделать вывод, что модель смогла найти определенные закономерности в данных

Пример сегментации листа из датасета street leaves:
![image](https://user-images.githubusercontent.com/85474856/136532726-384a13cb-a4de-4060-8023-b9e6cf12878e.png)


Наконец, в результате обучения модели на собственном датасете (street dataset) ошибка на тренировке составила 0.061, на валидации - 0.045, а Jaccard метрика - 0.809.

Примеры сегментации листа из датасета street leaves (больше примеров в [Demo.py](%TODO)):

![image](https://user-images.githubusercontent.com/85474856/136532931-18fadc5f-0beb-4bab-882c-423933c57e40.png)

![image](https://user-images.githubusercontent.com/85474856/136532970-ec7cb534-e43a-40d7-87e9-c040cf7b5968.png)






