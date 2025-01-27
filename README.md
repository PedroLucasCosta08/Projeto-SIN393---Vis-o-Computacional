# Classificação de pragas e doenças agrícolas por meio de imagens digitais utilizando aprendizado profundo.


Pedro Lucas de Oliveira Costa - 8127, Thiago Matheus de Oliveira Costa - 8101

Projeto em python desenvolvido para o treinamento de modelos de CNN sobre o dataset Crop Pest and Disease Detection utilizando o framework torchvision com
otimização de hyperparametros por meio do arquivo grid-search.py e teste integrado de multiplas estrategias de aumento de dados no arquivo train_test_batch.
Ao final da execução gera os resultados e analises  por matriz de confusão e metricas como, Acurácia, Precisão, Recall e F1-score.

## IMPORTANTE

O Dataset Crop Pest and Disease Detection possui arquivos defeituosos que impedem a execução dos experimentos, para isso desenvolvemos o script delDefectiveimages.py para excluir os arquivos defeituosos, rodar-lo antes de comecar os experimentos

## Desenvolvimento do Ambiente

* Install Anaconda from the Website https://www.anaconda.com/.

### Criando o Ambiente Conda
```
    $ conda update -n base -c defaults conda
    
    $ conda create -n env-CNNpestes-py310 python=3.10
    $ conda activate env-CNNpestes-py310

    $ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    $ conda install chardet
    $ pip install notebook
    $ pip install matplotlib
    $ pip install scikit-image
    $ pip install scikit-learn
    $ pip install pandas
    $ pip install seaborn
    $ pip install ipywidgets
    $ pip install ipympl
    $ pip install graphviz
    $ pip install opencv-python
    $ pip install albumentations
    $ pip install timm

```

### Salvar o Ambiente

```
    $ conda env export > env-CNNpestes-py310.yml
```

### Carregar ambiente por arquivo yml

```
    $ conda env create -f env-CNNpestes-py310.yml 
```

## Convertendo ipynb para py
---

```
    $ jupyter nbconvert CNNPestes.ipynb --to python
```


## Executando os Experimentos

###  Experimentos de hiperparâmetros

```
    $ nohup python grid-search.py --arch alexnet --optimizer Adam &
    $ nohup python grid-search.py --arch efficientnet_b4 --optimizer Adam &
    $ nohup python grid-search.py --arch mobilenet_v3 --optimizer Adam &
    $ nohup python grid-search.py --arch resnet50 --optimizer Adam &
```

* Escolha os melhores valores de hiperparametros

### Experimentos principais

* Execute cada experimento através do arquivo CNNpestes.py, ou...
* Defina os experimentos que serão executados no arquivo train_test_batch.py.

```
    $ nohup python train_test_batch.py
```

## Utilidades:
---

* Memória da GPU não libera, mesmo nvidia-smi não mostrando nenhum processo:
    * https://stackoverflow.com/a/46597252

```
fuser -v /dev/nvidia*
                     USER        PID ACCESS COMMAND
/dev/nvidia0:        joao      44525 F...m python
/dev/nvidiactl:      joao      44525 F...m python
/dev/nvidia-uvm:     joao      44525 F...m python

kill -9 44525
```

* List Python processes
```
    $ ps -ef | grep python
```
