import math
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from typing import Any, List, Tuple, Optional


class InkTrackAnalyzer:
    """
    Classe para análise automática do crescimento de uma mancha de tinta em vídeo.

    Esta classe encapsula todas as funcionalidades necessárias para:
    1. Carregar um vídeo.
    2. Permitir que o usuário calibre a escala espacial usando uma régua no vídeo.
    3. Permitir que o usuário selecione uma Região de Interesse (ROI) para análise.
    4. Permitir que o usuário calibre um limiar (threshold) para segmentar a mancha de tinta.
    5. Processar o vídeo frame a frame, detectando a mancha de tinta na ROI.
    6. Calcular o raio e a área da mancha em cada frame.
    7. Salvar os dados resultantes (tempo, raio, área) em um arquivo CSV.

    A interação com o usuário para calibração é realizada através de janelas gráficas do OpenCV.
    Os parâmetros de entrada e saída podem ser fornecidos via argumentos de linha de comando.
    """

    def __init__(self, video_path: str, output_csv_path: str):
        self.video_path: str = video_path
        self.output_csv_path: str = output_csv_path

        # As etapas da calibração agora seguem uma ordem lógica:
        # 0: Calibração da Régua
        # 1: Seleção da ROI
        # 2: Calibração do Threshold
        self.calibration_stage: int = -1  # Inicializa como -1, será 0 na primeira etapa

        # Calibração da escala
        self.drawing_points: List[Tuple[int, int]] = (
            []
        )  # Pontos para calibração da régua
        self.calibration_done: bool = (
            False  # Indica se a calibração da régua foi concluída
        )

        # Calibração do recorte
        self.roi_points: List[Tuple[int, int]] = (
            []
        )  # Pontos para seleção da área de corte (ROI)
        self.roi_selected: bool = False  # Indica se a seleção da ROI foi concluída

        # Calibração do limiar
        self.threshold_calibrated: bool = (
            False  # Indica se a calibração do threshold foi concluída
        )
        self.calibrated_threshold: int = 0  # Valor do threshold calibrado pelo usuário

        # self.start_frame_index: int = 0    # Removido: Índice do frame onde a análise realmente começa
        self.threshold_calibration_frame: Optional[np.ndarray] = (
            None  # Armazena o frame selecionado para calibração do threshold
        )

        self.pixels_per_unit: float = 0.0
        self.real_length: float = 10.0
        self.unit_name: str = "mm"
        self.data: List[List[float]] = []

        # Variáveis para armazenar as coordenadas da área de corte
        self.cropped_frame_offset_x: int = 0
        self.cropped_frame_offset_y: int = 0
        self.cropped_width: int = 0
        self.cropped_height: int = 0

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any):
        # A fase de calibração é definida antes de chamar o callback
        # para garantir que os pontos sejam capturados na fase correta.
        if self.calibration_stage == 0:  # Fase de calibração da régua (anteriormente 1)
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.drawing_points) < 2:
                    self.drawing_points.append((x, y))
                    print(f"Ponto {len(self.drawing_points)} selecionado: ({x}, {y})")
                    if len(self.drawing_points) == 2:
                        self.calibration_done = True  # Régua calibrada

        elif self.calibration_stage == 1:  # Fase de seleção da ROI (anteriormente 2)
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.roi_points) < 2:
                    self.roi_points.append((x, y))
                    print(f"Ponto ROI {len(self.roi_points)} selecionado: ({x}, {y})")
                    if len(self.roi_points) == 2:
                        self.roi_selected = True  # ROI selecionada
        # Não há callback de mouse para a fase de threshold, pois usa trackbar

    def _show_message(
        self,
        message: str,
        frame: np.ndarray,
        y: int = 60,
        color=(255, 255, 0),
        font_scale=2,
        thickness=3,
    ):
        """
        Escreve uma mensagem sobre o frame, no topo, de forma persistente.
        """
        cv2.putText(
            frame,
            message,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _calibrate_ruler(
        self, current_base_frame: np.ndarray, display_frame_for_drawing: np.ndarray
    ) -> bool:
        """
        Calibra a escala de pixels por unidade de medida real usando uma régua no frame base.
        Retorna True se a calibração for bem-sucedida, False caso contrário.
        """
        self.calibration_stage = 0
        self.drawing_points = []
        self.calibration_done = False

        while not self.calibration_done:
            temp_display_frame = current_base_frame.copy()
            self._show_message(
                "Clique em dois pontos da regua para calibrar a escala",
                temp_display_frame,
            )
            if len(self.drawing_points) == 1:
                cv2.circle(
                    temp_display_frame, self.drawing_points[0], 20, (0, 0, 255), -1
                )
            cv2.imshow("Calibrar", temp_display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False

        cv2.line(
            display_frame_for_drawing,
            self.drawing_points[0],
            self.drawing_points[1],
            (0, 255, 0),
            5,
        )
        cv2.circle(
            display_frame_for_drawing, self.drawing_points[0], 20, (0, 0, 255), -1
        )
        cv2.circle(
            display_frame_for_drawing, self.drawing_points[1], 20, (0, 0, 255), -1
        )
        cv2.imshow("Calibrar", display_frame_for_drawing)
        cv2.waitKey(1)

        p1 = self.drawing_points[0]
        p2 = self.drawing_points[1]
        pixel_distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        real_length = self.real_length
        input_digits = []
        while True:
            input_box = display_frame_for_drawing.copy()
            self._show_message(
                f"Digite o comprimento real da regua em {self.unit_name} e pressione ENTER: {''.join(map(str, input_digits))}",
                input_box,
            )
            cv2.imshow("Calibrar", input_box)

            key = cv2.waitKey(0)
            
            if key == 13:  # ENTER
                try:
                    real_length = float("".join(map(str, input_digits)))
                    if real_length > 0:
                        break
                except Exception:
                    pass

            elif 48 <= key <= 57:  # 0-9
                input_digits.append(key - 48)

            elif key in (8, 127):  # BACKSPACE
                if input_digits:
                    input_digits.pop()

            elif key == ord("q"):
                return False

        self.real_length = real_length
        self.pixels_per_unit = pixel_distance / real_length
        return True

    def _select_roi(
        self, current_base_frame: np.ndarray, display_frame_for_drawing: np.ndarray
    ) -> bool:
        """
        Permite ao usuário selecionar uma Região de Interesse (ROI) para análise.
        Retorna True se a seleção for bem-sucedida, False caso contrário.
        """
        self.calibration_stage = 1
        self.roi_points = []
        self.roi_selected = False
        while not self.roi_selected:
            temp_display_frame = current_base_frame.copy()
            self._show_message(
                "Selecione a ROI: clique no canto sup. esquerdo e inf. direito",
                temp_display_frame,
            )
            if len(self.roi_points) == 1:
                cv2.circle(
                    temp_display_frame, self.roi_points[0], 20, (0, 255, 255), -1
                )
            cv2.imshow("Calibrar", temp_display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        cv2.rectangle(display_frame_for_drawing, (x1, y1), (x2, y2), (0, 255, 255), 5)
        cv2.circle(display_frame_for_drawing, self.roi_points[0], 20, (0, 255, 255), -1)
        cv2.circle(display_frame_for_drawing, self.roi_points[1], 20, (0, 255, 255), -1)
        self._show_message(
            "ROI selecionada com sucesso!", display_frame_for_drawing, y=120
        )
        cv2.imshow("Calibrar", display_frame_for_drawing)
        cv2.waitKey(1)
        self.cropped_frame_offset_x = min(x1, x2)
        self.cropped_frame_offset_y = min(y1, y2)
        self.cropped_width = abs(x2 - x1)
        self.cropped_height = abs(y2 - y1)
        return True

    def _calibrate_threshold(self, current_base_frame_roi: np.ndarray) -> bool:
        """
        Permite ao usuário calibrar o valor do threshold para binarização da imagem,
        usando o frame base já cortado pela ROI.
        Retorna True se a calibração for bem-sucedida, False caso contrário.
        """
        self.calibration_stage = 2
        self.threshold_calibrated = False
        self.threshold_calibration_frame = current_base_frame_roi.copy()
        if (
            self.threshold_calibration_frame.shape[0] == 0
            or self.threshold_calibration_frame.shape[1] == 0
        ):
            return False
        gray_roi = cv2.cvtColor(self.threshold_calibration_frame, cv2.COLOR_BGR2GRAY)
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        def on_trackbar(val: int):
            pass

        cv2.createTrackbar("Threshold", "Calibrar", 0, 255, on_trackbar)
        cv2.setTrackbarPos("Threshold", "Calibrar", 127)
        while not self.threshold_calibrated:
            threshold_val = cv2.getTrackbarPos("Threshold", "Calibrar")
            _, binary_roi = cv2.threshold(
                blurred_roi, threshold_val, 255, cv2.THRESH_BINARY_INV
            )
            combined_threshold_display = np.hstack(
                (
                    self.threshold_calibration_frame,
                    cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR),
                )
            )
            self._show_message(
                "Ajuste o limiar na barra e pressione ENTER",
                combined_threshold_display,
                y=60,
            )

            cv2.imshow("Calibrar", combined_threshold_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                self.calibrated_threshold = threshold_val
                self.threshold_calibrated = True

            elif key == ord("q"):
                return False
            
        cv2.waitKey(500)
        return True

    def calibrate(self) -> bool:
        """
        Gerencia o fluxo completo de calibração (régua, ROI e threshold).
        Retorna True se todas as etapas de calibração forem bem-sucedidas, False caso contrário.
        """
        cap: cv2.VideoCapture = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(
                "Erro ao abrir o vídeo. Verifique o caminho ou se o codec está instalado."
            )
            return False

        cv2.namedWindow("Calibrar", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibrar", 1600, 900)
        cv2.setMouseCallback("Calibrar", self.mouse_callback)

        ret, frame = cap.read()
        if not ret:
            print(
                "Erro: Não foi possível ler o primeiro frame do vídeo para calibração."
            )
            cap.release()
            cv2.destroyAllWindows()
            return False

        # display_frame_for_drawing será o frame que acumula os desenhos de calibração (régua e ROI)
        # Ele começa como uma cópia do current_base_frame
        display_frame_for_drawing: np.ndarray = frame.copy()

        # --- FASE 0: CALIBRAÇÃO DA RÉGUA ---
        # Passa o frame base (primeiro frame) e o frame para desenho
        if not self._calibrate_ruler(frame, display_frame_for_drawing):
            cap.release()
            cv2.destroyAllWindows()
            return False

        # --- FASE 1: SELEÇÃO DA ÁREA DE CORTE (ROI) ---
        # Passa o frame base (primeiro frame) e o frame para desenho
        if not self._select_roi(frame, display_frame_for_drawing):
            cap.release()
            cv2.destroyAllWindows()
            return False

        # --- FASE 2: CALIBRAÇÃO DO THRESHOLD ---
        # Antes de chamar a calibração do threshold, cortamos o frame base pela ROI
        x_start: int = self.cropped_frame_offset_x
        y_start: int = self.cropped_frame_offset_y
        x_end: int = x_start + self.cropped_width
        y_end: int = y_start + self.cropped_height

        h_base, w_base, _ = frame.shape
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(w_base, x_end)
        y_end = min(h_base, y_end)

        current_base_frame_roi: np.ndarray = frame[y_start:y_end, x_start:x_end].copy()

        # Passa o frame base CORTADO pela ROI para a calibração do threshold
        if not self._calibrate_threshold(current_base_frame_roi):
            cap.release()
            cv2.destroyAllWindows()
            return False

        cv2.destroyWindow("Calibrar")
        cap.release()
        return True

    def process(self) -> None:
        if not self.calibrate():
            return

        cap: cv2.VideoCapture = cv2.VideoCapture(self.video_path)
        # A análise sempre começa do frame 0 agora
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fps: float = cap.get(cv2.CAP_PROP_FPS)

        # O frame_count agora representa o índice real do frame no vídeo
        actual_frame_count: int = 0

        cv2.namedWindow("Analise", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Analise", 1600, 900)

        print("\n--- INICIANDO ANÁLISE DO VÍDEO ---")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Aplica o corte (ROI) ao frame
            x_start: int = self.cropped_frame_offset_x
            y_start: int = self.cropped_frame_offset_y
            x_end: int = x_start + self.cropped_width
            y_end: int = y_start + self.cropped_height

            # Garante que as coordenadas da ROI estejam dentro dos limites do frame
            h, w, _ = frame.shape
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(w, x_end)
            y_end = min(h, y_end)

            # Realiza o corte
            cropped_frame: np.ndarray = frame[
                y_start:y_end, x_start:x_end
            ].copy()  # .copy() para garantir que seja um array gravável

            # Se o frame cortado estiver vazio (ex: ROI inválida), pula o processamento deste frame
            if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
                print(
                    f"Aviso: Frame {actual_frame_count} resultou em uma área de corte vazia. Pulando."
                )
                actual_frame_count += 1
                continue

            # Todo o processamento subsequente acontece no cropped_frame
            gray: np.ndarray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            blurred: np.ndarray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Usa o threshold calibrado pelo usuário
            _, binary = cv2.threshold(
                blurred, self.calibrated_threshold, 255, cv2.THRESH_BINARY_INV
            )

            kernel: np.ndarray = np.ones((3, 3), np.uint8)
            binary = cv2.erode(binary, kernel, iterations=1)
            binary = cv2.dilate(binary, kernel, iterations=1)

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            current_radius_pixels: int = 0
            current_area_pixels_sq: float = 0

            # Cria uma cópia do cropped_frame para desenhar o círculo, mantendo o original para a visualização combinada
            frame_with_circle: np.ndarray = cropped_frame.copy()

            if contours:
                max_contour: np.ndarray = max(contours, key=cv2.contourArea)
                current_area_pixels_sq: float = cv2.contourArea(max_contour)

                if len(max_contour) >= 5:
                    (x, y), radius_pixels = cv2.minEnclosingCircle(max_contour)
                    center: Tuple[int, int] = (int(x), int(y))
                    current_radius_pixels = int(radius_pixels)
                    cv2.circle(
                        frame_with_circle, center, current_radius_pixels, (0, 255, 0), 2
                    )
                    cv2.circle(
                        frame_with_circle, center, 10, (0, 255, 0), -1
                    )  # Desenha o centro da mancha

            real_world_radius: float = (
                current_radius_pixels / self.pixels_per_unit
                if self.pixels_per_unit > 0
                else 0
            )

            # Converte a área para unidades do mundo real
            real_world_area: float = (
                current_area_pixels_sq / (self.pixels_per_unit**2)
                if self.pixels_per_unit > 0
                else 0
            )

            # O tempo é calculado a partir do frame 0
            time_in_seconds: float = actual_frame_count / fps
            # Adiciona a área aos dados
            self.data.append([time_in_seconds, real_world_radius, real_world_area])

            # Prepara a imagem binária para ser concatenada (converte para BGR)
            binary_bgr: np.ndarray = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            # Concatena o frame com o círculo e o frame binarizado
            combined_frame: np.ndarray = np.hstack((frame_with_circle, binary_bgr))

            cv2.putText(
                combined_frame,
                f"Tempo: {time_in_seconds:.2f}s",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,  # Tamanho da fonte ajustado para caber
                (255, 255, 255),
                3,
            )

            cv2.putText(
                combined_frame,
                f"Raio: {real_world_radius:.2f} {self.unit_name}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,  # Tamanho da fonte ajustado para caber
                (255, 255, 255),
                3,
            )

            cv2.putText(
                combined_frame,
                f"Area: {real_world_area:.2f} {self.unit_name}^2",  # Texto para a área
                (10, 180),  # Posição abaixo do raio
                cv2.FONT_HERSHEY_SIMPLEX,
                2,  # Tamanho da fonte ajustado para caber
                (255, 255, 255),
                3,
            )

            cv2.imshow("Analise", combined_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Análise interrompida pelo usuário.")
                break
            actual_frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        self.save_data()

    def save_data(self) -> None:
        if not os.path.exists("data"):
            os.mkdir("data")

        df: pd.DataFrame = pd.DataFrame(
            self.data,
            columns=[
                "tempo",
                "raio",
                "area",
            ],
        )
        df.to_csv(self.output_csv_path, index=False)
        print(f"Dados salvos em: {self.output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Análise automática do crescimento de mancha de tinta em vídeo."
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Caminho para o vídeo a ser analisado.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Caminho para salvar o arquivo CSV de saída.",
    )
    # Adiciona argumento opcional para unidade de medida
    parser.add_argument(
        "--unit",
        type=str,
        default="mm",
        help="Unidade de medida (ex: mm, cm, px). Padrão: mm",
    )
    args = parser.parse_args()

    # Garante que o caminho de saída seja uma string válida
    output_path = args.output if args.output else None
    if output_path is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        output_dir = "data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, base + ".csv")

    analyzer = InkTrackAnalyzer(args.video, output_path)
    analyzer.unit_name = args.unit

    try:
        analyzer.process()
    except Exception as e:
        print(f"Erro inesperado durante o processamento: {e}")


if __name__ == "__main__":
    main()
