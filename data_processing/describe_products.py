import asyncio
from openai import AsyncOpenAI, APITimeoutError
from tqdm import tqdm
import pandas as pd
import os

client = AsyncOpenAI(
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai",
)

CONCURRENT_REQUESTS = 8
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

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

user_prompt = """Produto: {product_name}
Categoria: {category}
Subcategoria: {subcategory}
"""

async def describe_product(client, product, retries=3):
    try:
        async with semaphore:
            chat_completion = await client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-405B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt.format(
                            product_name=product["product_name"],
                            category=product["product_category"],
                            subcategory=product["product_subcategory"],
                        ),
                    },
                ],
                temperature=0.2,
                timeout=10, 
            )
        return chat_completion.choices[0].message.content
    except (APITimeoutError, asyncio.TimeoutError) as e:
        if retries > 0:
            print(f"Timeout occurred, retrying... ({retries} retries left)")
            return await describe_product(client, product, retries - 1)
        else:
            print(f"Failed to get a response for product {product['product_name']} after retries.")
            return "Description unavailable due to timeout"

async def describe_products(products: pd.DataFrame) -> pd.DataFrame:
    tasks = []
    products["description"] = None
    save_interval = 250
    save_counter = 0
    
    for index, product in products.iterrows():
        task = asyncio.create_task(describe_product(client, product))
        tasks.append((index, task))
    
    transformed_data_path = os.path.join(os.path.dirname(os.getcwd()), 'SJ_PCD_24-2', 'data', 'transformed')

    for index, task in tqdm(tasks):
        products.at[index, "description"] = await task
        save_counter += 1

        if save_counter % save_interval == 0:
            # Save the progress every 250 iterations
            temp_save_path = os.path.join(transformed_data_path, f'products_descriptions_temp_{save_counter}.parquet')
            products.to_parquet(temp_save_path, index=False)
            print(f"Progress saved at iteration {save_counter} to {temp_save_path}")

    # Final save after all products are processed
    final_save_path = os.path.join(transformed_data_path, 'products_descriptions.parquet')
    products.to_parquet(final_save_path, index=False)
    
    return products

def main():
    cleaned_data_path = os.path.join(os.path.dirname(os.getcwd()), 'SJ_PCD_24-2', 'data', 'cleaned')
    products = pd.read_parquet(os.path.join(cleaned_data_path, 'products.parquet'))
    
    # Run the asynchronous describe_products function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    products = loop.run_until_complete(describe_products(products))
    
    print(products.head())

if __name__ == "__main__":
    main()
