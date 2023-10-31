---
# Modelo Wave2Vec 2.0

O modelo Wave2Vec 2.0, desenvolvido pela MetaIA, é uma inovação na tecnologia de transcrição de áudio. Ele converte áudio em texto com base em uma arquitetura de transformers, sendo treinado em uma extensa base de dados que abrange inúmeras línguas ao redor do mundo. Este modelo foi treinado com aproximadamente 5.000 línguas e dialetos, prometendo ser uma opção mais leve e aprimorada em relação ao Whisper da OpenAI.

---

### Implementação

O código fornecido com este README contém um arquivo `code.py` que permite ajustar o Wave2Vec 2.0 com base na arquitetura Transformers. A base de treinamento usada é a conhecida base de dados do Common Voice da Mozilla Firefox, que contém dados etiquetados, incluindo áudio e texto correspondente.

---

### Sobre o Código

Este código utiliza a métrica WER (Word Error Rate) para avaliar o erro médio na transcrição do áudio. O código é bastante abrangente e inclui funções para acessar a base de dados, bem como métricas de treinamento. Você pode personalizar as métricas conforme necessário. Se desejar um modelo de maior qualidade, é recomendável ajustar os checkpoints, o número de épocas e o tamanho do lote (batch size) para um treinamento mais aprofundado. Aumentar a base de dados e diversificar os dados também pode melhorar a métrica de WER, especialmente para vozes menos comuns.

---
### Instruções

Para executar o código como está, basta inserir o seguinte comando no terminal:
``` bash
python wave2vec.py
```

---
### Autor

**Rafael Rosendo Faustino**

- Aluno de Tecnologia da Informação na Universidade Federal do Rio Grande do Norte.
- Bolsista de pesquisa em processamento de fala com modelos de linguagem de grande escala.

---

Certifique-se de que o seu código Python esteja contido em um arquivo chamado `wave2vec.py` para que o comando acima funcione corretamente.

