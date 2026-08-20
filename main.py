"""
Olho Amigo - Detector de Objetos com Áudio
TCC Técnico em Informática - IFPB (2024)

Detecta objetos em tempo real via câmera e descreve em áudio,
promovendo acessibilidade para pessoas com deficiência visual.

Código original adaptado e organizado mantendo a lógica de detecção.
"""

import cv2
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import os
import pygame
import time

# Lista de classes COCO
NOMES_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Dicionário de tradução para português
NOMES_CLASSES_TRADUZIDO = {
    "person": "pessoa", "bicycle": "bicicleta", "car": "carro", "motorcycle": "motocicleta", 
    "airplane": "avião", "bus": "ônibus", "train": "trem", "truck": "caminhão", "boat": "navio",
    "traffic light": "semáforo", "fire hydrant": "hidrante", "stop sign": "placa de pare", 
    "parking meter": "parquímetro", "bench": "banco", "bird": "pássaro", "cat": "gato",
    "dog": "cachorro", "horse": "cavalo", "sheep": "ovelha", "cow": "vaca", "elephant": "elefante", 
    "bear": "urso", "zebra": "zebra", "giraffe": "girafa", "backpack": "mochila", "umbrella": "guarda-chuva",
    "handbag": "bolsa de mão", "tie": "gravata", "suitcase": "mala", "frisbee": "frisbee", "skis": "esquis", 
    "snowboard": "snowboard", "sports ball": "bola de esportes", "kite": "pipa", "baseball bat": "taco de baseball",
    "baseball glove": "luva de baseball", "skateboard": "skate", "surfboard": "prancha de surf", 
    "tennis racket": "raquete de tênis", "bottle": "garrafa", "wine glass": "taça de vinho", "cup": "copo",
    "fork": "garfo", "knife": "faca", "spoon": "colher", "bowl": "tigela", "banana": "banana", 
    "apple": "maçã", "sandwich": "sanduíche", "orange": "laranja", "broccoli": "brócolis",
    "carrot": "cenoura", "hot dog": "cachorro quente", "pizza": "pizza", "donut": "donut", "cake": "bolo", 
    "chair": "cadeira", "couch": "sofá", "potted plant": "planta", "bed": "cama",
    "dining table": "mesa de jantar", "toilet": "banheiro", "tv": "tv", "laptop": "notebook", 
    "mouse": "mouse", "remote": "controle remoto", "keyboard": "teclado", "cell phone": "celular",
    "microwave": "microondas", "oven": "forno", "toaster": "torradeira", "sink": "pia", 
    "refrigerator": "geladeira", "book": "livro", "clock": "relógio", "vase": "vaso", "scissors": "tesoura",
    "teddy bear": "urso de pelúcia", "hair drier": "secador", "toothbrush": "escova de dente"
}


class SistemaAudio:
    """Gerencia reprodução de áudio com pygame"""
    
    def __init__(self):
        """Inicializa o sistema de áudio"""
        pygame.mixer.init()
    
    def reproduzir_nome_objeto(self, nome_classe):
        """Reproduz o nome do objeto em áudio"""
        try:
            # Criar arquivo de áudio temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as arquivo_audio_temp:
                caminho_arquivo_audio_temp = arquivo_audio_temp.name
                
                # Gerar áudio com gTTS
                tts = gTTS(text=nome_classe, lang='pt')
                tts.save(caminho_arquivo_audio_temp)
            
            # Aguardar um pouco
            time.sleep(1)
            
            # Reproduzir se arquivo existe
            if os.path.exists(caminho_arquivo_audio_temp):
                pygame.mixer.music.load(caminho_arquivo_audio_temp)
                pygame.mixer.music.play()
        
        except Exception as e:
            print(f"Erro ao reproduzir áudio: {e}")


class DetectorObjetos:
    """Detector de objetos com YOLO"""
    
    def __init__(self):
        """Inicializa o detector YOLO"""
        print("🚀 Carregando modelo YOLO...")
        self.modelo = YOLO("yolov8n.pt")
        print("✅ Modelo carregado!\n")
        
        self.audio = SistemaAudio()
    
    def detectar(self, frame):
        """Detecta objetos no frame e retorna lista de detecções"""
        deteccoes = []
        resultados = self.modelo(frame, stream=True)
        
        for r in resultados:
            caixas = r.boxes
            for caixa in caixas:
                # Extrair coordenadas
                x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                cx = (x2 + x1) // 2
                cy = (y2 + y1) // 2
                
                confianca = round(caixa.conf[0].item(), 2)
                cls = int(caixa.cls[0])
                
                # Obter nome traduzido
                nome_classe_en = NOMES_CLASSES[cls]
                nome_classe_pt = NOMES_CLASSES_TRADUZIDO.get(nome_classe_en, "Desconhecido")
                
                # Adicionar detecção
                deteccoes.append({
                    'bbox': (x1, y1, x2, y2),
                    'centro': (cx, cy),
                    'confianca': confianca,
                    'nome': nome_classe_pt,
                    'nome_en': nome_classe_en
                })
        
        return deteccoes
    
    def desenhar_deteccoes(self, frame, deteccoes):
        """Desenha retângulos e labels no frame"""
        for det in deteccoes:
            x1, y1, x2, y2 = det['bbox']
            nome = det['nome']
            confianca = det['confianca']
            
            # Desenhar retângulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Preparar label
            rotulo = f'{nome} {confianca}'
            tamanho_texto = cv2.getTextSize(rotulo, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + tamanho_texto[0], y1 - tamanho_texto[1] - 3
            
            # Desenhar fundo do label
            cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1, cv2.LINE_AA)
            
            # Desenhar texto
            cv2.putText(frame, rotulo, (x1, y1 - 2), 0, 1, [255, 255, 255], 
                       thickness=1, lineType=cv2.LINE_AA)
        
        return frame


class CameraTempoReal:
    """Controla a câmera e processamento em tempo real"""
    
    def __init__(self):
        """Inicializa a câmera"""
        self.cam = cv2.VideoCapture(0)
        
        if not self.cam.isOpened():
            print("❌ Erro ao abrir a câmera!")
            exit()
        
        self.detector = DetectorObjetos()
        self.contador = 0
    
    def processar(self):
        """Processa frames da câmera continuamente"""
        print("📷 Câmera iniciada")
        print("⌨️  Pressione '1' para sair\n")
        
        while self.cam.isOpened():
            self.contador += 1
            ret, frame = self.cam.read()
            
            if not ret:
                break
            
            # Detectar objetos
            deteccoes = self.detector.detectar(frame)
            
            # Reproduzir áudio para cada detecção
            for det in deteccoes:
                self.detector.audio.reproduzir_nome_objeto(det['nome'])
            
            # Desenhar detecções
            frame = self.detector.desenhar_deteccoes(frame, deteccoes)
            
            # Redimensionar para exibição
            frame_redimensionado = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6, 
                                            interpolation=cv2.INTER_AREA)
            
            # Mostrar frame
            cv2.imshow("👁️ Olho Amigo", frame_redimensionado)
            
            # Sair se pressionar '1'
            if cv2.waitKey(1) & 0xFF == ord('1'):
                print("\n👋 Encerrando...")
                break
    
    def finalizar(self):
        """Fecha câmera e janelas"""
        self.cam.release()
        cv2.destroyAllWindows()
        print("✅ Câmera fechada")


def main():
    """Função principal"""
    print("""

   👁️ OLHO AMIGO - CÂMERA TEMPO REAL

    """)
    
    camera = CameraTempoReal()
    
    try:
        camera.processar()
    except KeyboardInterrupt:
        print("\n👋 Interrompido pelo usuário")
    finally:
        camera.finalizar()


if __name__ == "__main__":
    main()
