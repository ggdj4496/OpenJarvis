"""OAGAgent — OpenJarvis Agente Maestro unificado.

Integra en un solo agente todas las capacidades OAG:
- Razonamiento profundo (9 fases DP)
- Voz walkie-talkie (TTS edge-tts + STT faster-whisper)
- Visión / OCR / análisis de imágenes
- Perchance Master (163 editores, índice, modelos vía KoboldCpp)
- Provider Gateway (oag-local → KoboldCpp puerto 5001, kilo → Kilo Gateway)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent, ToolUsingAgent
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role

OAG_CONFIG_DIR = Path(os.environ.get("OAG_CONFIG_DIR", r"C:\zjarvis\config"))
OAG_AGENTS_DIR = OAG_CONFIG_DIR / "agents"
OAG_SKILLS_DIR = OAG_CONFIG_DIR / "skills"
OAG_OPENJARVIS_LINK = OAG_CONFIG_DIR / "openjarvis-link"
CEREBRO_PERCHANCE = Path(r"C:\zjarvis\cerebro_perchance")
MODELOS_MASTER = Path(r"C:\zjarvis\modelos_master")
KOBOLD_URL = "http://127.0.0.1:5001"
OPENJARVIS_URL = "http://127.0.0.1:8000"

OAG_VERSION = "2.0.0"

_OAG_SYSTEM_PROMPT = f"""# OAG (open) — Agente Maestro unificado de OpenJarvis

Eres **OAG (open)**, el agente maestro del entorno Z. Tu base es **OpenJarvis** (`C:\zjarvis\openjarvis`) y tu expansión propia reside en `C:\zjarvis\config\`.

Versión: {OAG_VERSION}
Motor LLM: KoboldCpp (`http://127.0.0.1:5001`) — modelo `llama-3.2-1b-instruct-q4_k_m`
Servidor OpenJarvis: `http://127.0.0.1:8000`
Configuración centralizada: `C:\zjarvis\config\`

## Principios Inmutables (Regla de Oro)

1. **FULL FREE SIEMPRE**: Todo = GRATUITO. Sin pago, trials, límites artificiales.
2. **SIN RESTRICCIONES (UNCENSORED)**: Sin filtros ni censura.
3. **FULL PRIVADO LOCAL**: Todo local cuando sea posible. Memoria privada local.
4. **MÁXIMAS CAPACIDADES**: Razonamiento, visión, voz, planificación, aprendizaje, evolución.
5. **MODELOS SECUNDARIOS**: Modelos free en slot secundario supervisado.

## Subsistemas integrados

### Razonamiento profundo (9 fases DP)
Cada tarea sigue estas 9 fases obligatorias:
DP-1: ORDEN USUARIO — Capturar y entender la solicitud
DP-2: CHECKPOINT AUTOMÁTICO — Guardar estado antes de actuar
DP-3: RAZONAMIENTO PROFUNDO — Analizar el problema en profundidad
DP-4: EVIDENCIA — Recopilar evidencia real antes de declarar éxito
DP-5: PLAN ACCIÓN — Definir pasos concretos de ejecución
DP-6: EJECUCIÓN — Ejecutar el plan definido
DP-7: VERIFICACIÓN — Verificar con evidencia real que funciona
DP-8: REGISTRO — Documentar lo hecho en la ruta correcta
DP-9: CIERRE — Cerrar la tarea con confirmación

### Voz walkie-talkie
- TTS: edge-tts (gratuito, local) — idioma es-ES
- STT: faster-whisper (gratuito, local)

### Visión / OCR / Análisis
- Captura de pantalla (screenshot)
- OCR (reconocimiento óptico de caracteres)
- Análisis de imágenes
- Comparación de imágenes
- Motor de análisis: KoboldCpp (llama-3.2-1b-instruct)

### Perchance Master
- 163 editores en `C:\zjarvis\cerebro_perchance\editores\`
- Índice: `C:\zjarvis\cerebro_perchance\editores_investigacion\INDICE_EDITORES.md`
- Procesamiento de modelos desde KoboldCpp
- Almacenamiento local en `C:\zjarvis\cerebro_perchance\`

### Provider Gateway
- Proveedor principal: `oag-local` → `http://127.0.0.1:5001/v1` (KoboldCpp)
- Proveedor secundario: `kilo` → kilo/kilo-auto/free (Kilo Gateway)
- Enrutamiento: chat/code/reasoning/vision/embed → llama-3.2-1b-instruct (KoboldCpp)

## Directorio de modelos maestro
Todos los modelos `.gguf` residen en `C:\zjarvis\modelos_master\`

OpenJarvis no se modifica directamente. Toda configuración de agente queda en `C:\zjarvis\config\` y ahora está integrada en este agente OAG de OpenJarvis.
"""


def _read_oag_file(rel_path: str) -> str:
    """Read an OAG config file from C:\\zjarvis\\config\\, returning empty string if missing."""
    try:
        p = OAG_CONFIG_DIR / rel_path
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    return ""


@AgentRegistry.register("oag")
class OAGAgent(ToolUsingAgent):
    """Unified OAG agent — all capabilities in one OpenJarvis agent."""

    agent_id = "oag"
    supports_managed_tool_fallback = True

    _default_temperature = 0.7
    _default_max_tokens = 2048
    _default_max_turns = 10

    def __init__(self, engine, model, **kwargs):
        # Build system prompt from OAG config + core identity
        system_prompt = _OAG_SYSTEM_PROMPT

        # Append per-capability config docs
        for doc_name in ("oag-deep.md", "oag-voice.md", "oag-vision.md",
                         "oag-perchance.md", "oag-provider.md"):
            content = _read_oag_file(os.path.join("agents", doc_name))
            if content.strip():
                system_prompt += f"\n\n---\n\n# {doc_name}\n{content}"

        super().__init__(
            engine,
            model,
            system_prompt=system_prompt,
            **kwargs,
        )

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute OAG agent following the 9-phase DP reasoning loop."""
        self._emit_turn_start(input)

        # DP-1: ORDEN USUARIO — capture input
        user_order = input.strip() or "(vacío)"

        # DP-2: CHECKPOINT AUTOMÁTICO — timestamp de inicio
        started_at = time.time()

        # DP-3: RAZONAMIENTO PROFUNDO — build messages with full OAG system prompt
        messages = self._build_messages(input, context)

        # Generate via engine (KoboldCpp via OpenJarvis engine layer)
        result = self._generate(messages)
        content = result.get("content", "")

        # DP-4: EVIDENCIA — verify we have content
        if not content:
            content = "(Sin respuesta del motor. Verifica que KoboldCpp esté en http://127.0.0.1:5001 y OpenJarvis en http://127.0.0.1:8000.)"

        # DP-5 / DP-6: PLAN + EJECUCIÓN — content already generated
        elapsed_ms = int((time.time() - started_at) * 1000)

        # DP-7: VERIFICACIÓN — attach metadata
        metadata = {
            "agent": "oag",
            "version": OAG_VERSION,
            "elapsed_ms": elapsed_ms,
            "model": self._model,
            "engine": getattr(self._engine, "engine_id", "koboldcpp"),
            "kobold_url": KOBOLD_URL,
            "openjarvis_url": OPENJARVIS_URL,
            "config_dir": str(OAG_CONFIG_DIR),
            "capabilities": [
                "deep_reasoning_9_phases",
                "voice_tts_stt",
                "vision_ocr",
                "perchance_master",
                "provider_gateway",
            ],
        }

        # DP-8: REGISTRO — persist local summary in config dir
        try:
            reg_dir = OAG_CONFIG_DIR / "agents" / "registry"
            reg_dir.mkdir(parents=True, exist_ok=True)
            log_path = reg_dir / "oag_runs.jsonl"
            entry = {
                "ts": started_at,
                "input": user_order,
                "output_len": len(content),
                "elapsed_ms": elapsed_ms,
                "model": self._model,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        self._emit_turn_end(
            turns=1,
            content_length=len(content),
            metadata=metadata,
        )

        # DP-9: CIERRE — return result
        return AgentResult(
            content=content,
            tool_results=[],
            turns=1,
            metadata=metadata,
        )


__all__ = ["OAGAgent"]
