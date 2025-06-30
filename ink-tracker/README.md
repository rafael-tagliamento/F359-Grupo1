# InkTrackAnalyzer

Este projeto realiza a análise automática do crescimento de uma mancha de tinta em vídeo, utilizando OpenCV e Python. O usuário calibra a escala e seleciona a região de interesse (ROI) no vídeo, e o script extrai o raio e a área da mancha ao longo do tempo, salvando os dados em um arquivo CSV.

Este algoritmo foi desenvolvido por estudantes da Unicamp para um experimento de Física na disciplina F 359. O objetivo do experimento é determinar o coeficiente de difusão \( D \) e analisar como ele varia com a temperatura da água. Para isso, utiliza-se a solução da equação diferencial de Bessel para descrever a concentração de tinta ao longo do tempo, permitindo extrair \( D \) a partir dos dados obtidos no vídeo.

## Como funciona

O script guia o usuário através de um processo interativo de calibração usando janelas do OpenCV:

1.  **Calibração da régua:**
    -   O usuário clica em dois pontos sobre uma régua (ou objeto de tamanho conhecido) visível no primeiro frame do vídeo.
    -   Em seguida, digita o comprimento real correspondente diretamente na janela do OpenCV e pressiona ENTER. Isso define a escala (pixels por unidade real, ex: pixels/mm).
    -   É possível usar a tecla BACKSPACE para corrigir a entrada e 'q' para cancelar a calibração.
2.  **Seleção da ROI (Região de Interesse):**
    -   O usuário clica em dois pontos (canto superior esquerdo e inferior direito) para definir a área retangular do vídeo onde a mancha será analisada.
    -   Pressione 'q' para cancelar.
3.  **Calibração do threshold (Limiar):**
    -   Uma janela exibe o frame da ROI e sua versão binarizada lado a lado.
    -   O usuário ajusta uma trackbar (controle deslizante) para encontrar o limiar ideal que segmenta a mancha de tinta do fundo.
    -   Pressione ENTER para confirmar o threshold ou 'q' para cancelar.
4.  **Processamento:**
    -   Após a calibração bem-sucedida, o script processa o vídeo frame a frame dentro da ROI definida.
    -   Para cada frame, ele converte para escala de cinza, aplica um desfoque Gaussiano para suavizar ruídos, binariza a imagem usando o threshold calibrado e aplica operações de erosão e dilatação para refinar a segmentação da mancha.
    -   O maior contorno encontrado é considerado a mancha de tinta.
    -   O raio (baseado no círculo mínimo envolvente) e a área do contorno são calculados em unidades reais.
    -   Os dados (tempo, raio, área) são armazenados.
    -   Durante o processamento, uma janela exibe o frame da ROI com a mancha detectada e informações de tempo, raio e área. Pressione 'q' para interromper a análise.
5.  **Salvamento dos Dados:**
    -   Ao final do processamento (ou interrupção), os dados coletados são salvos em um arquivo CSV.

## Requisitos

-   Python 3.8+
-   [uv](https://github.com/astral-sh/uv) (gerenciador de dependências ultrarrápido)

## Instalação das dependências

Com a ferramenta [uv](https://github.com/astral-sh/uv) instalada, basta rodar:

```bash
uv sync
```

Isso instalará as dependências listadas em `pyproject.toml` (como OpenCV, NumPy, Pandas).

## Como usar

1.  Certifique-se de que suas dependências estão instaladas (`uv sync`).
2.  Coloque o vídeo a ser analisado na pasta `videos/` (opcional, você pode especificar qualquer caminho).
3.  Execute o script principal através do terminal. Abaixo alguns exemplos:

    -   **Comando básico (vídeo obrigatório):**

        ```bash
        uv run track.py --video caminho/para/seu/video.mp4
        ```

        _Neste caso, o arquivo de saída CSV será salvo automaticamente na pasta `data/` com o mesmo nome do vídeo (ex: `data/video.csv`). A unidade de medida padrão será "mm"._

    -   **Especificando arquivo de saída e unidade:**
        ```bash
        uv run track.py --video videos/meu_experimento.MOV --output resultados/experimento_final.csv --unit cm
        ```

    **Argumentos disponíveis:**

    -   `--video CAMINHO_VIDEO` (obrigatório): Caminho para o arquivo de vídeo de entrada.
    -   `--output CAMINHO_CSV` (opcional): Caminho completo para salvar o arquivo CSV de saída. Se não fornecido, o padrão é `data/NOME_DO_VIDEO.csv`.
    -   `--unit NOME_UNIDADE` (opcional): Unidade de medida para a calibração e resultados (ex: mm, cm, m, in). Padrão: `mm`.

4.  Siga as instruções interativas nas janelas do OpenCV para calibrar (régua, ROI, threshold) e depois aguarde o processamento do vídeo.
5.  Os resultados serão salvos no local especificado (ou padrão). O arquivo CSV conterá as colunas: `tempo`, `raio`, `area`.

## Estrutura do projeto

-   `track.py` — Script principal de análise.
-   `videos/` — Coloque aqui seus vídeos.
-   `data/` — Resultados em CSV serão salvos aqui.
-   `README.md` — Este arquivo.
-   `pyproject.toml` e `uv.lock` — Gerenciamento de dependências com uv.

## Observações

-   O script requer ambiente gráfico para exibir as janelas do OpenCV.
-   Para vídeos grandes, o processamento pode demorar.
-   O código é orientado a objetos e fácil de adaptar para outros experimentos.

---

Desenvolvido para fins acadêmicos (disciplina F 359) na Unicamp.
