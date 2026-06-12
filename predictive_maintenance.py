pythonimport numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. SIMULAÇÃO DE DADOS HISTÓRICOS INDUSTRIAIS (Indústria 4.0)
print("⚙️ Gerando base de dados simulada de telemetria industrial...")
np.random.seed(42)
n_registros = 1000

# Criando variáveis comuns de chão de fábrica
dados = {
    'temperatura_ar_K': np.random.uniform(295, 305, n_registros),
    'temperatura_processo_K': np.random.uniform(305, 315, n_registros),
    'velocidade_rotacao_rpm': np.random.uniform(1300, 2800, n_registros),
    'torque_Nm': np.random.uniform(10, 70, n_registros),
    'desgaste_ferramenta_min': np.random.uniform(0, 250, n_registros)
}

df = pd.DataFrame(dados)

# Criando uma lógica de falha baseada nas variáveis (regras de engenharia)
# Se a temperatura subir muito ou o torque/desgaste passarem do limite, a chance de falha (1) aumenta
condicao_falha = (
    (df['temperatura_processo_K'] > 312) & (df['torque_Nm'] > 55) |
    (df['desgaste_ferramenta_min'] > 220) & (df['velocidade_rotacao_rpm'] > 2400)
)
df['falha'] = np.where(condicao_falha, 1, 0)

# Garantindo que temos amostras de falhas suficientes na simulação
amostras_falha = df[df['falha'] == 1].shape[0]
print(f"📊 Total de registros gerados: {n_registros} | Falhas mecânicas identificadas: {amostras_falha}\n")

# 2. DIVISÃO DOS DADOS (Treinamento e Teste)
X = df.drop(columns=['falha'])  # Características/Variáveis de entrada
y = df['falha']                 # Alvo/Target (0 = Normal, 1 = Falha Mecânica)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.3, random_state=42, stratify=y)

# 3. TREINAMENTO DO MODELO (Machine Learning)
print("🔮 Treinando o modelo preditivo (Random Forest Classifier)...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
print("✅ Modelo treinado com sucesso!\n")

# 4. AVALIAÇÃO DE PERFORMANCE PARA APRESENTAÇÃO
y_pred = modelo.predict(X_test)

print("====== RELATÓRIO DE MÉTRICAS OPERACIONAIS ======")
print(f"Acurácia Geral do Sistema: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Matriz de Confusão (Previsão vs Real):")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório Técnico Detalhado:")
print(classification_report(y_test, y_pred, target_names=['Operação Normal', 'Falha Detectada']))

# 5. ANÁLISE DE IMPORTÂNCIA DE ATIVOS (Quais variáveis causam mais falhas?)
importâncias = modelo.feature_importances_
print("====== IMPACTO DE VARIÁVEIS NA FALHA ======")
for var, imp in zip(X.columns, importâncias):
    print(f"• {var}: {imp * 100:.2f}% de impacto preditivo")
