# TODO - Agregar Generación de Gráficos/Visuales con Gemini

## Tareas Pendientes
- [x] Instalar bibliotecas necesarias (matplotlib, pillow) si no están presentes
- [x] Agregar endpoint /generate-visual en icfes_api.py
- [x] Implementar lógica para generar gráficos de barras/pie para análisis de preguntas
- [x] Integrar Gemini para generar prompts o datos para visuales
- [x] Probar el endpoint y verificar que genere imágenes correctamente
- [ ] Actualizar frontends HTML para consumir el nuevo endpoint si es necesario

## Detalles del Plan
- **Objetivo**: Permitir que Gemini genere gráficos visuales relacionados con preguntas ICFES (ej. estadísticas de respuestas, diagramas).
- **Implementación**: Usar Matplotlib para crear gráficos y devolver como imagen base64.
- **Endpoint**: POST /generate-visual con parámetros como tipo de gráfico y datos.
- **Dependencias**: matplotlib, pillow (para manejo de imágenes).
