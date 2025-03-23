from django.shortcuts import render, redirect, HttpResponse
from .models import dados
import os
from django.core.files.storage import FileSystemStorage
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import numpy as np



def ia_import(request):
    return render(request, 'ia_import.html')

def ia_import_save(request):
    if request.method == 'POST' and request.FILES['arq_upload']:
        fss = FileSystemStorage()
        upload = request.FILES['arq_upload']
        file1 = fss.save(upload.name, upload)
        file_url = fss.url(file1)
        
        dados.objects.all().delete()
        
        i = 0
        file2 = open(file1, 'r')
        
        for row in file2:
            if (i > 0):
                #cells = row.split(',')
                cells = re.split(',|;', row)

                def safe_int(value, default=0):
                    return int(value) if value.strip() else default
                
                dados.objects.create(
                    NU_ANO_CENSO = safe_int(cells[0]), NO_REGIAO=cells[1].strip() if cells[1].strip() else "-",
                    CO_REGIAO = safe_int(cells[2]), NO_UF=cells[3].strip() if cells[3].strip() else "-",
                    SG_UF=cells[4].strip() if cells[4].strip() else "-",
                    CO_UF = safe_int(cells[5]), NO_MUNICIPIO=cells[6].strip() if cells[6].strip() else "-",
                    CO_MUNICIPIO = safe_int(cells[7]), IN_CAPITAL = safe_int(cells[8]), TP_DIMENSAO = safe_int(cells[9]),
                    TP_ORGANIZACAO_ACADEMICA = safe_int(cells[10]), TP_REDE = safe_int(cells[11]), TP_CATEGORIA_ADMINISTRATIVA = safe_int(cells[12]),
                    CO_IES = safe_int(cells[13]), NO_CURSO=cells[14].strip() if cells[14].strip() else "-",
                    CO_CURSO = safe_int(cells[15]), NO_CINE_ROTULO = cells[16].strip() if cells[16].strip() else "-",
                    CO_CINE_ROTULO = cells[17].strip() if cells[17].strip() else "-", CO_CINE_AREA_GERAL = cells[18].strip() if cells[18].strip() else "-",
                    NO_CINE_AREA_GERAL = cells[19].strip() if cells[19].strip() else "-", CO_CINE_AREA_ESPECIFICA = cells[20].strip() if cells[20].strip() else "-",
                    NO_CINE_AREA_ESPECIFICA = cells[21].strip() if cells[21].strip() else "-", CO_CINE_AREA_DETALHADA = cells[22].strip() if cells[22].strip() else "-",
                    NO_CINE_AREA_DETALHADA = cells[23].strip() if cells[23].strip() else "-",
                    TP_GRAU_ACADEMICO = safe_int(cells[24]), IN_GRATUITO = safe_int(cells[25]), TP_MODALIDADE_ENSINO = safe_int(cells[26]), TP_NIVEL_ACADEMICO = safe_int(cells[27]),
                    QT_CURSO = safe_int(cells[28]), QT_VG_TOTAL = safe_int(cells[29]), QT_VG_TOTAL_DIURNO = safe_int(cells[30]), QT_VG_TOTAL_NOTURNO = safe_int(cells[31]),
                    QT_VG_TOTAL_EAD = safe_int(cells[32]), QT_VG_NOVA = safe_int(cells[33]), QT_VG_PROC_SELETIVO = safe_int(cells[34]), QT_VG_REMANESC = safe_int(cells[35]),
                    QT_VG_PROG_ESPECIAL = safe_int(cells[36]), QT_INSCRITO_TOTAL = safe_int(cells[37]), QT_INSCRITO_TOTAL_DIURNO = safe_int(cells[38]),
                    QT_INSCRITO_TOTAL_NOTURNO = safe_int(cells[39]), QT_INSCRITO_TOTAL_EAD = safe_int(cells[40]), QT_INSC_VG_NOVA = safe_int(cells[41]),
                    QT_INSC_PROC_SELETIVO = safe_int(cells[42]), QT_INSC_VG_REMANESC = safe_int(cells[43]), QT_INSC_VG_PROG_ESPECIAL = safe_int(cells[44]),
                    QT_ING = safe_int(cells[45]), QT_ING_FEM = safe_int(cells[46]), QT_ING_MASC = safe_int(cells[46]), QT_ING_DIURNO = safe_int(cells[48]),
                    QT_ING_NOTURNO = safe_int(cells[49]), QT_ING_VG_NOVA = safe_int(cells[50]), QT_ING_VESTIBULAR = safe_int(cells[51]), QT_ING_ENEM = safe_int(cells[52]),
                    QT_ING_AVALIACAO_SERIADA = safe_int(cells[53]), QT_ING_SELECAO_SIMPLIFICA = safe_int(cells[54]), QT_ING_EGR = safe_int(cells[55]),
                    QT_ING_OUTRO_TIPO_SELECAO = safe_int(cells[56]), QT_ING_PROC_SELETIVO = safe_int(cells[57]), QT_ING_VG_REMANESC = safe_int(cells[58]),
                    QT_ING_VG_PROG_ESPECIAL = safe_int(cells[59]), QT_ING_OUTRA_FORMA = safe_int(cells[60]), QT_ING_0_17 = safe_int(cells[61]), QT_ING_18_24 = safe_int(cells[62]),
                    QT_ING_25_29 = safe_int(cells[63]), QT_ING_30_34 = safe_int(cells[64]), QT_ING_35_39 = safe_int(cells[65]), QT_ING_40_49 = safe_int(cells[66]),
                    QT_ING_50_59 = safe_int(cells[67]), QT_ING_60_MAIS = safe_int(cells[68]), QT_ING_BRANCA = safe_int(cells[69]), QT_ING_PRETA = safe_int(cells[70]),
                    QT_ING_PARDA = safe_int(cells[71]), QT_ING_AMARELA = safe_int(cells[72]), QT_ING_INDIGENA = safe_int(cells[73]), QT_ING_CORND = safe_int(cells[74]),
                    QT_MAT = safe_int(cells[75]), QT_MAT_FEM = safe_int(cells[76]), QT_MAT_MASC = safe_int(cells[77]), QT_MAT_DIURNO = safe_int(cells[78]),
                    QT_MAT_NOTURNO = safe_int(cells[79]), QT_MAT_0_17 = safe_int(cells[80]), QT_MAT_18_24 = safe_int(cells[81]), QT_MAT_25_29 = safe_int(cells[82]),
                    QT_MAT_30_34 = safe_int(cells[83]), QT_MAT_35_39 = safe_int(cells[84]), QT_MAT_40_49 = safe_int(cells[85]), QT_MAT_50_59 = safe_int(cells[86]),
                    QT_MAT_60_MAIS = safe_int(cells[87]), QT_MAT_BRANCA = safe_int(cells[88]), QT_MAT_PRETA = safe_int(cells[89]), QT_MAT_PARDA = safe_int(cells[90]),
                    QT_MAT_AMARELA = safe_int(cells[91]), QT_MAT_INDIGENA = safe_int(cells[92]), QT_MAT_CORND = safe_int(cells[93]), QT_CONC = safe_int(cells[94]),
                    QT_CONC_FEM = safe_int(cells[95]), QT_CONC_MASC = safe_int(cells[96]), QT_CONC_DIURNO = safe_int(cells[97]), QT_CONC_NOTURNO = safe_int(cells[98]),
                    QT_CONC_0_17 = safe_int(cells[99]), QT_CONC_18_24 = safe_int(cells[100]), QT_CONC_25_29 = safe_int(cells[101]), QT_CONC_30_34 = safe_int(cells[102]),
                    QT_CONC_35_39 = safe_int(cells[103]), QT_CONC_40_49 = safe_int(cells[104]), QT_CONC_50_59 = safe_int(cells[105]), QT_CONC_60_MAIS = safe_int(cells[106]),
                    QT_CONC_BRANCA = safe_int(cells[107]), QT_CONC_PRETA = safe_int(cells[108]), QT_CONC_PARDA = safe_int(cells[109]), QT_CONC_AMARELA = safe_int(cells[110]),
                    QT_CONC_INDIGENA = safe_int(cells[111]), QT_CONC_CORND = safe_int(cells[112]), QT_ING_NACBRAS = safe_int(cells[113]), QT_ING_NACESTRANG = safe_int(cells[114]),
                    QT_MAT_NACBRAS = safe_int(cells[115]), QT_MAT_NACESTRANG = safe_int(cells[116]), QT_CONC_NACBRAS = safe_int(cells[117]), QT_CONC_NACESTRANG = safe_int(cells[118]),
                    QT_ALUNO_DEFICIENTE = safe_int(cells[119]), QT_ING_DEFICIENTE = safe_int(cells[120]), QT_MAT_DEFICIENTE = safe_int(cells[121]), QT_CONC_DEFICIENTE = safe_int(cells[122]),
                    QT_ING_FINANC = safe_int(cells[123]), QT_ING_FINANC_REEMB = safe_int(cells[124]), QT_ING_FIES = safe_int(cells[125]), QT_ING_RPFIES = safe_int(cells[126]),
                    QT_ING_FINANC_REEMB_OUTROS = safe_int(cells[127]), QT_ING_FINANC_NREEMB = safe_int(cells[128]), QT_ING_PROUNII = safe_int(cells[129]),
                    QT_ING_PROUNIP = safe_int(cells[130]), QT_ING_NRPFIES = safe_int(cells[131]), QT_ING_FINANC_NREEMB_OUTROS = safe_int(cells[132]), QT_MAT_FINANC = safe_int(cells[133]),
                    QT_MAT_FINANC_REEMB = safe_int(cells[134]), QT_MAT_FIES = safe_int(cells[135]), QT_MAT_RPFIES = safe_int(cells[136]), QT_MAT_FINANC_REEMB_OUTROS = safe_int(cells[137]),
                    QT_MAT_FINANC_NREEMB = safe_int(cells[138]), QT_MAT_PROUNII = safe_int(cells[139]), QT_MAT_PROUNIP = safe_int(cells[140]), QT_MAT_NRPFIES = safe_int(cells[141]),
                    QT_MAT_FINANC_NREEMB_OUTROS = safe_int(cells[142]), QT_CONC_FINANC = safe_int(cells[143]), QT_CONC_FINANC_REEMB = safe_int(cells[144]), QT_CONC_FIES = safe_int(cells[145]),
                    QT_CONC_RPFIES = safe_int(cells[146]), QT_CONC_FINANC_REEMB_OUTROS = safe_int(cells[147]), QT_CONC_FINANC_NREEMB = safe_int(cells[148]), QT_CONC_PROUNII = safe_int(cells[149]),
                    QT_CONC_PROUNIP = safe_int(cells[150]), QT_CONC_NRPFIES = safe_int(cells[151]), QT_CONC_FINANC_NREEMB_OUTROS = safe_int(cells[152]), QT_ING_RESERVA_VAGA = safe_int(cells[153]),
                    QT_ING_RVREDEPUBLICA = safe_int(cells[154]), QT_ING_RVETNICO = safe_int(cells[155]), QT_ING_RVPDEF = safe_int(cells[156]), QT_ING_RVSOCIAL_RF = safe_int(cells[157]),
                    QT_ING_RVOUTROS = safe_int(cells[158]), QT_MAT_RESERVA_VAGA = safe_int(cells[159]), QT_MAT_RVREDEPUBLICA = safe_int(cells[160]), QT_MAT_RVETNICO = safe_int(cells[161]),
                    QT_MAT_RVPDEF = safe_int(cells[162]), QT_MAT_RVSOCIAL_RF = safe_int(cells[163]), QT_MAT_RVOUTROS = safe_int(cells[164]), QT_CONC_RESERVA_VAGA = safe_int(cells[165]),
                    QT_CONC_RVREDEPUBLICA = safe_int(cells[166]), QT_CONC_RVETNICO = safe_int(cells[167]), QT_CONC_RVPDEF = safe_int(cells[168]), QT_CONC_RVSOCIAL_RF = safe_int(cells[169]),
                    QT_CONC_RVOUTROS = safe_int(cells[170]), QT_SIT_TRANCADA = safe_int(cells[171]), QT_SIT_DESVINCULADO = safe_int(cells[172]), QT_SIT_TRANSFERIDO = safe_int(cells[173]),
                    QT_SIT_FALECIDO = safe_int(cells[174]), QT_ING_PROCESCPUBLICA = safe_int(cells[175]), QT_ING_PROCESCPRIVADA = safe_int(cells[176]),
                    QT_ING_PROCNAOINFORMADA = safe_int(cells[177]), QT_MAT_PROCESCPUBLICA = safe_int(cells[178]), QT_MAT_PROCESCPRIVADA = safe_int(cells[179]),
                    QT_MAT_PROCNAOINFORMADA = safe_int(cells[180]), QT_CONC_PROCESCPUBLICA = safe_int(cells[181]), QT_CONC_PROCESCPRIVADA = safe_int(cells[182]),
                    QT_CONC_PROCNAOINFORMADA = safe_int(cells[183]), QT_PARFOR = safe_int(cells[184]), QT_ING_PARFOR = safe_int(cells[185]), QT_MAT_PARFOR = safe_int(cells[186]),
                    QT_CONC_PARFOR = safe_int(cells[187]), QT_APOIO_SOCIAL = safe_int(cells[188]), QT_ING_APOIO_SOCIAL = safe_int(cells[189]), QT_MAT_APOIO_SOCIAL = safe_int(cells[190]),
                    QT_CONC_APOIO_SOCIAL = safe_int(cells[191]), QT_ATIV_EXTRACURRICULAR = safe_int(cells[192]), QT_ING_ATIV_EXTRACURRICULAR = safe_int(cells[193]),
                    QT_MAT_ATIV_EXTRACURRICULAR = safe_int(cells[194]), QT_CONC_ATIV_EXTRACURRICULAR = safe_int(cells[195]), QT_MOB_ACADEMICA = safe_int(cells[196]),
                    QT_ING_MOB_ACADEMICA = safe_int(cells[197]), QT_MAT_MOB_ACADEMICA = safe_int(cells[198]), QT_CONC_MOB_ACADEMICA = safe_int(cells[199])
                )
                
            i = i+1
        file2.close()
        os.remove(file_url.replace("/", ""))
    return redirect('ia_import_list')

def ia_import_list(request):
    data = {}
    data['dados'] = dados.objects.all()
    return render(request, 'ia_import_list.html', data)

def ia_knn_treino(request):
    data = {}
    print("Vamos ao que interessa...")

    dados_queryset = dados.objects.all()
    print("Registros Selecionados.")

    df = pd.DataFrame(list(dados_queryset.values()))
    print("Pandas Carregado e dados 'convertidos'.")
    print("'Cabeçalho' dos dados:")
    print(df.head())

    df.fillna(-1, inplace=True)

    def categorizar_qt_ing(valor):
        if valor < 50:
            return 0
        elif 50 <= valor < 100:
            return 1
        else:
            return 2

    df['QT_ING_CATEGORIA'] = df['QT_ING'].apply(categorizar_qt_ing)

    categorical_columns = ['NO_REGIAO', 'NO_UF', 'SG_UF', 'NO_MUNICIPIO', 'NO_CURSO',
                           'NO_CINE_ROTULO', 'CO_CINE_ROTULO', 'CO_CINE_AREA_GERAL', 
                           'NO_CINE_AREA_GERAL', 'CO_CINE_AREA_ESPECIFICA', 
                           'NO_CINE_AREA_ESPECIFICA', 'CO_CINE_AREA_DETALHADA', 
                           'NO_CINE_AREA_DETALHADA']

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    x = df.drop(columns=['QT_ING', 'QT_ING_CATEGORIA', 'id'])
    y = df['QT_ING_CATEGORIA']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    data['dataset'] = x_train.shape
    data['treino'] = x_train.shape[0]
    data['teste'] = x_test.shape[0]
    data['validacao'] = x_val.shape[0]

    print(f'Tamanho do conjunto de treino: {x_train.shape}')
    print(f'Tamanho do conjunto de teste: {x_test.shape}')
    print(f'Tamanho do conjunto de validação: {x_val.shape}')

    knn = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric':  ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    data['best'] = grid_search.best_params_
    print("Melhores parâmetros encontrados:", grid_search.best_params_)

    best_knn = grid_search.best_estimator_

    y_val_pred = best_knn.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Acurácia no conjunto de validação: {val_accuracy * 100:.2f}%')
    data['acc_validacao'] = round(val_accuracy * 100, 2)

    y_test_pred = best_knn.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    data['acc_teste'] = round(test_accuracy * 100, 2)
    print(f'Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%')

    model_filename = 'knn_model.pkl'
    model_data = {'model': best_knn, 'encoders': label_encoders}
    joblib.dump(model_data, model_filename)
    print(f'Modelo salvo em: {model_filename}')
    data['file'] = model_filename

    return render(request, 'ia_knn_treino.html', data)

def ia_knn_matriz(request):
    dados_queryset = dados.objects.all()
    df = pd.DataFrame(list(dados_queryset.values()))

    def categorizar_qt_ing(valor):
        if valor < 50:
            return 0
        elif 50 <= valor < 100:
            return 1
        else:
            return 2

    df['QT_ING_CATEGORIA'] = df['QT_ING'].apply(categorizar_qt_ing)

    model_filename = 'knn_model.pkl'
    model_data = joblib.load(model_filename)
    best_knn = model_data['model']
    label_encoders = model_data['encoders']

    print("Modelo carregado com sucesso.")

    categorical_columns = ['NO_REGIAO', 'NO_UF', 'SG_UF', 'NO_MUNICIPIO', 'NO_CURSO',
                           'NO_CINE_ROTULO', 'CO_CINE_ROTULO', 'CO_CINE_AREA_GERAL', 
                           'NO_CINE_AREA_GERAL', 'CO_CINE_AREA_ESPECIFICA', 
                           'NO_CINE_AREA_ESPECIFICA', 'CO_CINE_AREA_DETALHADA', 
                           'NO_CINE_AREA_DETALHADA']
    
    for col in categorical_columns:
        if col in df.columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            else:
                return HttpResponse(f"Erro: LabelEncoder para {col} não encontrado no modelo.")

    x = df.drop(columns=['QT_ING', 'QT_ING_CATEGORIA', 'id'])
    y = df['QT_ING_CATEGORIA']

    y_pred = best_knn.predict(x)
    cm = confusion_matrix(y, y_pred)
    data = {
        'matrix': cm.tolist(),
        'labels': np.unique(y).tolist()
    }
    for i in data['matrix']:
        print(i)
    return render(request, 'ia_knn_matriz.html', data)