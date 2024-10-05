import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai",
)


# Define data root path and read the products CSV file
data_root = Path("../data")

# Create path if it does not exist
if not data_root.exists():
    data_root.mkdir()

if not (data_root / "transformed").exists():
    (data_root / "transformed").mkdir()


products = pd.read_csv(data_root / "cleaned/products.csv")


# Discard 'price' and 'product_id' columns
# NOTE: I should not have done this. but since I have already done it, I will just leave it like that
products = products.drop(columns=["price", "product_id"])


# Define the system prompt for generating descriptions
system_prompt = """**Prompt para geração de descrições:**

"Para cada produto listado, crie uma descrição técnica, neutra e detalhada que inclua:

- **Nome do produto**: Especifique o nome completo do produto.
- **Categoria**: Indique a categoria à qual o produto pertence.
- **Subcategoria**: Mencione a subcategoria do produto.
- **Descrição**: Forneça uma explicação clara, detalhada e neutra sobre o uso do produto, destacando suas principais características e indicações. A descrição deve incluir os seguintes pontos:
  1. **Para que o produto é utilizado** (beneficio, finalidade ou tratamento específico).
  2. **Composição principal** (ingredientes ativos, se relevante).
  
As descrições devem ser objetivas, focadas nas aplicações e propriedades do produto, sem adicionar informações promocionais ou exageros.


**As saidas devem estar unica e exclusivamente no formato abaixo:**

- **Nome do produto**: Curitybina 5ml União Química
- **Categoria**: Medicamentos
- **Subcategoria**: Similar
- **Descrição**: A Curitybina 5ml da União Química é um colírio antibiótico indicado para o tratamento de infecções oculares, como conjuntivite bacteriana e outras inflamações causadas por microrganismos sensíveis ao antibiótico presente na fórmula. Sua ação antimicrobiana atua diretamente sobre as bactérias, ajudando a eliminar a infecção e a reduzir os sintomas associados, como vermelhidão, dor e secreção ocular. A Curitybina deve ser aplicada diretamente nos olhos, conforme orientação médica, com doses ajustadas à gravidade da infecção. Recomenda-se o uso conforme prescrição para evitar resistência bacteriana.


- **Nome do produto**: Algodão Cremer 50g Rolo
- **Categoria**: Medicamentos
- **Subcategoria**: Hospitalares
- **Descrição**: O Algodão Cremer 50g em rolo é um material hospitalar de uso geral amplamente utilizado em procedimentos de assepsia, curativos e cuidados com a pele. Composto por fibras 100% algodão hidrófilo, apresenta alta capacidade de absorção, sendo ideal para limpeza de feridas, aplicação de medicamentos tópicos e cuidados pós-cirúrgicos. O produto também é indicado para uso em ambiente doméstico, auxiliando na remoção de maquiagem, higiene de bebês e cuidados com a pele sensível. Seu formato em rolo permite fácil manuseio e corte conforme a necessidade do procedimento.

- **Nome do produto**: Anador 500mg 4 Comprimidos Opella
- **Categoria**: Medicamentos
- **Subcategoria**: Referência Avulso
- **Descrição**: Anador 500mg da Opella é um medicamento analgésico e antipirético que contém dipirona sódica monoidratada como princípio ativo. É indicado para o alívio de dores de intensidade leve a moderada, como dores de cabeça, dores musculares, cólicas e dores pós-operatórias, além de ser eficaz na redução da febre. Anador atua inibindo a produção de substâncias responsáveis pela dor e febre no organismo, oferecendo alívio rápido e temporário dos sintomas. A dosagem recomendada é de 500mg, e o uso deve ser orientado por um médico, especialmente em casos de hipersensibilidade ou histórico de reações adversas a analgésicos.

- **Nome do produto**: Condicionador Seda 325ml Cachos
- **Categoria**: Perfumaria
- **Subcategoria**: Perfumaria
- **Descrição**: O Condicionador Seda Cachos 325ml foi formulado especialmente para cabelos cacheados e ondulados, proporcionando hidratação intensa e definição dos cachos. Sua fórmula contém ingredientes como óleo de argan e proteínas que ajudam a nutrir profundamente os fios, prevenindo o ressecamento e o frizz. O uso regular promove cachos mais definidos, macios e brilhantes, além de facilitar o desembaraço dos cabelos, evitando a quebra. Ideal para manter a forma natural dos cachos, o produto é indicado para todos os tipos de curvatura, desde ondulados até cabelos crespos, podendo ser utilizado diariamente.

- **Nome do produto**: Bepantol Derma Solução 50ml Bayer
- **Categoria**: Perfumaria
- **Subcategoria**: Perfumaria
- **Descrição**: Bepantol Derma Solução 50ml da Bayer é um produto dermatológico multifuncional indicado para hidratação profunda da pele e dos cabelos. Sua fórmula contém dexpantenol (pró-vitamina B5), que auxilia no processo de regeneração celular e manutenção da hidratação natural. Quando aplicado na pele, Bepantol Derma melhora a elasticidade e a textura, sendo especialmente eficaz em áreas secas e sensíveis, como lábios, cotovelos e áreas pós-depilação. Nos cabelos, promove hidratação intensa, brilho e maciez, podendo ser aplicado diretamente nos fios ou misturado a outros produtos capilares. Ideal para uso diário e recomendado para todos os tipos de pele e cabelo.
"""


# Define the user prompt template
user_prompt = """Produto: {product_name}
Categoria: {category}
Subcategoria: {subcategory}
"""

# Initialize the columns
products["description"] = None
products["embedding"] = None

# Generate descriptions for each product
for index, product in tqdm(products.iterrows()):
    chat_completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt.format(
                    product_name=product["product_name"],
                    category=product["category"],
                    subcategory=product["subcategory"],
                ),
            },
        ],
        temperature=0.2,
    )
    content = chat_completion.choices[0].message.content
    products.at[index, "description"] = content

    # Save intermediate results every 250 iterations
    if index % 250 == 0:  # type: ignore
        products.to_csv(data_root / f"descriptions_{index}.csv", index=False)


# Load the SentenceTransformer model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Generate embeddings for each product description
for index, product in tqdm(products.iterrows()):
    embedding = model.encode(product["description"])
    products.at[index, "embedding"] = embedding

# Save the final DataFrame to a CSV file
products.to_csv(data_root / "cleaned/products_descriptions_embeddings.csv", index=False)
