# predictive-maintenance-ml
Modelo de Machine Learning em Python para previsão de falhas mecânicas e manutenção preditiva na Indústria 4.0.

-----------------------------------------------------//----------------------------------------------------------

markdown# 🔮 Predictive Maintenance ML

Uma solução de Inteligência Artificial desenvolvida em Python para previsão de falhas em ativos industriais, aplicando conceitos de Machine Learning para viabilizar estratégias de Manutenção Preditiva na Indústria 4.0.

---

### 📋 Sobre o Projeto
Este projeto utiliza algoritmos de Machine Learning para analisar variáveis de processos industriais (como temperatura de operação, velocidade de rotação, torque e desgaste de ferramentas) com o objetivo de prever se uma máquina sofrerá uma falha mecânica antes mesmo que ela ocorra, evitando paradas não planejadas e reduzindo o downtime.

### 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 🐍
- **Análise & Modelagem:** Scikit-Learn, Pandas, NumPy
- **Visualização de Dados:** Matplotlib, Seaborn
- **Ambiente:** Jupyter Notebook 📓

---

### 📊 Conjunto de Dados & Variáveis analisadas
O modelo avalia dados de telemetria das máquinas contendo:
- **Temperatura do Ar e do Processo [K]**
- **Velocidade de Rotação [rpm]**
- **Torque [Nm]**
- **Desgaste da Ferramenta [min]**
- **Tipo de Falha (Alvo do Modelo):** Falha por sobrecarga, falha por desgaste, falha de energia, entre outras.

---

### 🚀 Como Executar o Projeto
1. Clone o repositório:
   ```bash
   git clone https://github.com
   ```
2. Instale as bibliotecas necessárias:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn notebook
   ```
3. Inicie o Jupyter Notebook para executar o treinamento do modelo:
   ```bash
   jupyter notebook

   
