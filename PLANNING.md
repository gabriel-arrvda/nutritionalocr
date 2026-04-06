https://www.kaggle.com/datasets/mariogemoll/nutrition-facts
https://www.kaggle.com/datasets/shensivam/nutritional-facts-from-food-label
https://www.kaggle.com/datasets/gheysar4real/iranian-nutritional-fact-label

Baseado nesse dataset, o projeto de OCR (Optical Character Recognition) tem como objetivo desenvolver um sistema capaz de extrair informações nutricionais de imagens de rótulos de alimentos. O sistema deve ser capaz de identificar e extrair dados como calorias, carboidratos, proteínas, gorduras, vitaminas e minerais presentes nos rótulos.
Para alcançar esse objetivo, o projeto será dividido em várias etapas:
1. Coleta de Dados: Reunir um conjunto diversificado de imagens de rótulos de alimentos, garantindo que haja uma variedade de produtos e formatos de rótulos.
2. Pré-processamento de Imagens: Aplicar técnicas de pré-processamento para melhorar a qualidade das imagens, como ajuste de contraste, remoção de ruído e correção de perspectiva.
3. Treinamento do Modelo OCR: Utilizar um modelo de OCR, como Tesseract ou um modelo baseado em redes neurais, para treinar o sistema a reconhecer e extrair texto das imagens.
4. Extração de Informações Nutricionais: Desenvolver um algoritmo para identificar e extrair as informações nutricionais relevantes dos textos extraídos, organizando-os em um formato estruturado.
5. Validação e Testes: Avaliar a precisão do sistema utilizando um conjunto de dados de teste, comparando os resultados extraídos com as informações reais dos rótulos.
6. Implementação de Interface: Criar uma interface de usuário para facilitar a utilização do sistema, permitindo que os usuários façam upload de imagens de rótulos e visualizem as informações nutricionais extraídas de forma clara e organizada.
7. Otimização e Melhoria Contínua: Analisar os resultados e feedback dos usuários para identificar áreas de melhoria, otimizando o modelo e o algoritmo de extração para aumentar a precisão e a eficiência do sistema.
Ao final do projeto, o sistema de OCR deve ser capaz de fornecer informações nutricionais precisas e confiáveis a partir de imagens de rótulos de alimentos, contribuindo para a conscientização nutricional e auxiliando os consumidores a fazer escolhas alimentares mais informadas.

Importante: os rótulos estão em diversos idiomas, então o sistema deve ser capaz de lidar com múltiplos idiomas para garantir a extração correta das informações nutricionais.

Na primeira etapa na coleta de dados use pyhton e arquivo jupyter notebook para facilitar leitura e entendimento do processo. Utilize bibliotecas como pandas para manipulação de dados, requests para baixar imagens e BeautifulSoup para extrair links de imagens dos datasets mencionados. Certifique-se de organizar os dados coletados em um formato estruturado, como um DataFrame, para facilitar as etapas subsequentes do projeto.

Após isso faça uma api em nodejs ou pyhton para receber uma imagem e extrair os dados nutricionais utilizando o modelo de OCR treinado. A API deve ser capaz de receber uma imagem, processá-la e retornar as informações nutricionais extraídas em um formato JSON, facilitando a integração com outras aplicações ou interfaces de usuário.

Por últimmo faça um frontend angular para permitir que os usuários façam upload de imagens de rótulos de alimentos e visualizem as informações nutricionais extraídas. O frontend deve ser intuitivo e responsivo, proporcionando uma experiência de usuário agradável. Ele deve se comunicar com a API para enviar as imagens e receber os dados nutricionais, exibindo-os de forma clara e organizada para os usuários.