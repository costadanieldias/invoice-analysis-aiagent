#!/usr/bin/env python3
"""
Agente de IA para Análise de Notas Fiscais CSV
Utiliza LangChain + Pandas para análise inteligente de dados de notas fiscais
"""

import os
import zipfile
import pandas as pd
import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import tempfile
from pathlib import Path

class NotasFiscaisAnalyzer:
    def __init__(self, openai_api_key=None):
        """
        Inicializa o analisador de notas fiscais
        
        Args:
            openai_api_key: Chave da API OpenAI (opcional, pode ser definida via env)
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY deve ser fornecida via parâmetro ou variável de ambiente")
        
        self.df_cabecalho = None
        self.df_itens = None
        self.agent = None
        
        # Configurar LLM - Usando modelo mais acessível
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",  # Modelo mais acessível
            openai_api_key=self.openai_api_key
        )
    
    def extract_zip_files(self, zip_path):
        """
        Extrai arquivos CSV do ZIP fornecido
        
        Args:
            zip_path: Caminho para o arquivo ZIP
        """
        temp_dir = tempfile.mkdtemp()
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Procurar pelos arquivos CSV específicos
        cabecalho_path = None
        itens_path = None
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if 'Cabecalho' in file and file.endswith('.csv'):
                    cabecalho_path = os.path.join(root, file)
                elif 'Itens' in file and file.endswith('.csv'):
                    itens_path = os.path.join(root, file)
        
        if not cabecalho_path or not itens_path:
            raise FileNotFoundError("Arquivos CSV de cabeçalho e/ou itens não encontrados no ZIP")
        
        return cabecalho_path, itens_path
    
    def load_data(self, zip_path):
        """
        Carrega dados dos arquivos CSV extraídos do ZIP
        
        Args:
            zip_path: Caminho para o arquivo ZIP contendo os CSVs
        """
        try:
            # Extrair arquivos
            cabecalho_path, itens_path = self.extract_zip_files(zip_path)
            
            # Carregar DataFrames
            self.df_cabecalho = pd.read_csv(
                cabecalho_path,
                sep=',',
                decimal='.',
                parse_dates=True,
                encoding='utf-8'
            )
            
            self.df_itens = pd.read_csv(
                itens_path,
                sep=',',
                decimal='.',
                parse_dates=True,
                encoding='utf-8'
            )
            
            # Processar datas se necessário
            self._process_dates()
            
            # Criar agente LangChain
            self._create_agent()
            
            print(f"✅ Dados carregados com sucesso!")
            print(f"📊 Cabeçalho: {len(self.df_cabecalho)} registros")
            print(f"📊 Itens: {len(self.df_itens)} registros")
            
        except Exception as e:
            raise Exception(f"Erro ao carregar dados: {str(e)}")
    
    def _process_dates(self):
        """Processa colunas de data nos DataFrames"""
        date_columns = []
        
        # Identificar colunas de data no cabeçalho
        for col in self.df_cabecalho.columns:
            if 'data' in col.lower() or 'date' in col.lower():
                try:
                    self.df_cabecalho[col] = pd.to_datetime(self.df_cabecalho[col])
                    date_columns.append(col)
                except:
                    pass
        
        # Identificar colunas de data nos itens
        for col in self.df_itens.columns:
            if 'data' in col.lower() or 'date' in col.lower():
                try:
                    self.df_itens[col] = pd.to_datetime(self.df_itens[col])
                    date_columns.append(col)
                except:
                    pass
        
        if date_columns:
            print(f"📅 Colunas de data processadas: {date_columns}")
    
    def _create_agent(self):
        """Cria o agente LangChain para análise dos DataFrames"""
        
        # Contexto personalizado para notas fiscais
        prefix = """
        Você é um especialista em análise de dados de notas fiscais brasileiras.
        Você tem acesso a dois DataFrames:
        - df_cabecalho: Contém informações do cabeçalho das notas fiscais
        - df_itens: Contém os itens individuais das notas fiscais
        
        Quando responder perguntas:
        1. Seja preciso e objetivo
        2. Use formatação brasileira para números (vírgula como separador decimal)
        3. Para valores monetários, use o formato R$ X.XXX,XX
        4. Para datas, use formato DD/MM/AAAA
        5. Sempre explique brevemente como chegou ao resultado
        6. Se precisar fazer JOIN entre as tabelas, use colunas de chave comuns
        
        Exemplos de perguntas típicas:
        - "Qual fornecedor teve maior montante recebido?"
        - "Qual item teve maior volume entregue?"
        - "Qual a média de valor por nota fiscal?"
        """
        
        # Criar agente com ambos os DataFrames
        self.agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=[self.df_cabecalho, self.df_itens],
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=prefix,
            allow_dangerous_code=True  # Necessário para operações pandas
        )
    
    def query(self, question):
        """
        Executa uma pergunta no agente
        
        Args:
            question: Pergunta sobre os dados das notas fiscais
            
        Returns:
            Resposta do agente
        """
        if not self.agent:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        try:
            response = self.agent.run(question)
            return response
        except Exception as e:
            return f"Erro ao processar pergunta: {str(e)}"
    
    def get_data_summary(self):
        """Retorna resumo dos dados carregados"""
        if self.df_cabecalho is None or self.df_itens is None:
            return "Nenhum dado carregado."
        
        summary = f"""
        📊 **RESUMO DOS DADOS**
        
        **Cabeçalho das Notas Fiscais:**
        - Registros: {len(self.df_cabecalho):,}
        - Colunas: {len(self.df_cabecalho.columns)}
        - Colunas: {', '.join(self.df_cabecalho.columns.tolist())}
        
        **Itens das Notas Fiscais:**
        - Registros: {len(self.df_itens):,}
        - Colunas: {len(self.df_itens.columns)}
        - Colunas: {', '.join(self.df_itens.columns.tolist())}
        """
        
        return summary

def create_streamlit_interface():
    """Cria interface Streamlit para o agente"""
    
    st.set_page_config(
        page_title="Analisador de Notas Fiscais",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🤖 Agente IA - Análise de Notas Fiscais")
    st.markdown("---")
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Input para API Key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Insira sua chave da API OpenAI"
        )
        
        # Upload do arquivo ZIP
        uploaded_file = st.file_uploader(
            "📁 Upload do arquivo ZIP",
            type=['zip'],
            help="Faça upload do arquivo 202401_NFs.zip"
        )
    
    # Inicializar session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Carregar dados
    if uploaded_file and api_key and not st.session_state.data_loaded:
        try:
            with st.spinner("🔄 Carregando e processando dados..."):
                # Salvar arquivo temporário
                temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
                temp_path.write(uploaded_file.read())
                temp_path.close()
                
                # Inicializar analisador
                st.session_state.analyzer = NotasFiscaisAnalyzer(api_key)
                st.session_state.analyzer.load_data(temp_path.name)
                st.session_state.data_loaded = True
                
                # Limpar arquivo temporário
                os.unlink(temp_path.name)
                
            st.success("✅ Dados carregados com sucesso!")
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {str(e)}")
    
    # Interface principal
    if st.session_state.data_loaded:
        
        # Mostrar resumo dos dados
        with st.expander("📊 Resumo dos Dados", expanded=False):
            st.text(st.session_state.analyzer.get_data_summary())
        
        # Exemplos de perguntas
        st.subheader("💡 Exemplos de Perguntas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            example_questions = [
                "Qual fornecedor teve maior montante recebido?",
                "Qual item teve maior volume entregue?",
                "Qual a média de valor por nota fiscal?",
                "Quantas notas fiscais foram emitidas?",
                "Qual o valor total de todas as notas?"
            ]
            
            for i, question in enumerate(example_questions):
                if st.button(f"📝 {question}", key=f"example_{i}"):
                    st.session_state.current_question = question
        
        # Input para pergunta personalizada
        st.subheader("❓ Sua Pergunta")
        
        user_question = st.text_area(
            "Digite sua pergunta sobre as notas fiscais:",
            value=st.session_state.get('current_question', ''),
            height=100
        )
        
        if st.button("🔍 Analisar", type="primary"):
            if user_question.strip():
                with st.spinner("🤔 Analisando dados..."):
                    try:
                        response = st.session_state.analyzer.query(user_question)
                        
                        st.subheader("📋 Resposta")
                        st.write(response)
                        
                    except Exception as e:
                        st.error(f"❌ Erro na análise: {str(e)}")
            else:
                st.warning("⚠️ Por favor, digite uma pergunta.")
    
    else:
        # Instruções iniciais
        st.info("""
        📋 **Para começar:**
        
        1. Insira sua **OpenAI API Key** na barra lateral
        2. Faça **upload do arquivo ZIP** contendo os CSVs das notas fiscais
        3. Aguarde o processamento dos dados
        4. Faça suas perguntas sobre os dados!
        
        ✨ **Exemplos de perguntas que você pode fazer:**
        - Qual fornecedor recebeu mais dinheiro?
        - Qual produto teve maior quantidade vendida?
        - Qual a média de valor das notas fiscais?
        - Quantas notas foram emitidas por mês?
        """)

# Função principal para executar
def main():
    """Função principal - pode ser usada via linha de comando ou Streamlit"""
    
    # Verificar se está rodando no Streamlit
    try:
        import streamlit as st
        create_streamlit_interface()
    except:
        # Modo console
        print("🤖 Agente IA - Análise de Notas Fiscais")
        print("=" * 50)
        
        # Exemplo de uso programático
        api_key = input("OpenAI API Key: ")
        zip_path = input("Caminho para o arquivo ZIP: ")
        
        try:
            analyzer = NotasFiscaisAnalyzer(api_key)
            analyzer.load_data(zip_path)
            
            print("\n" + analyzer.get_data_summary())
            
            while True:
                question = input("\n❓ Sua pergunta (ou 'sair' para terminar): ")
                if question.lower() in ['sair', 'exit', 'quit']:
                    break
                
                print("🤔 Analisando...")
                response = analyzer.query(question)
                print(f"\n📋 Resposta: {response}")
                
        except Exception as e:
            print(f"❌ Erro: {str(e)}")

if __name__ == "__main__":
    main()
