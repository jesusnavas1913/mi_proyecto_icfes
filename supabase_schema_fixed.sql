-- ===========================================
-- ICFES Pro - Esquema de Base de Datos Supabase
-- ===========================================

-- ===========================================
-- TABLA: usuarios
-- ===========================================
CREATE TABLE IF NOT EXISTS usuarios (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    cedula VARCHAR(20) UNIQUE NOT NULL,
    rol VARCHAR(20) NOT NULL CHECK (rol IN ('estudiante', 'profesor')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Habilitar RLS en usuarios
ALTER TABLE usuarios ENABLE ROW LEVEL SECURITY;

-- Políticas RLS para usuarios
CREATE POLICY "Users can view their own profile" ON usuarios
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON usuarios
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Allow insert for authenticated users" ON usuarios
    FOR INSERT WITH CHECK (auth.uid() = id);

-- ===========================================
-- TABLA: examenes
-- ===========================================
CREATE TABLE IF NOT EXISTS examenes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    titulo VARCHAR(255) NOT NULL,
    descripcion TEXT,
    materia VARCHAR(100) NOT NULL,
    competencia VARCHAR(255) NOT NULL,
    dificultad VARCHAR(20) NOT NULL CHECK (dificultad IN ('facil', 'medio', 'avanzado')),
    numero_preguntas INTEGER NOT NULL DEFAULT 10,
    tiempo_limite INTEGER, -- en minutos
    profesor_id UUID REFERENCES usuarios(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Habilitar RLS en examenes
ALTER TABLE examenes ENABLE ROW LEVEL SECURITY;

-- Políticas RLS para examenes
CREATE POLICY "Profesores can view their own exams" ON examenes
    FOR SELECT USING (auth.uid() = profesor_id);

CREATE POLICY "Profesores can create exams" ON examenes
    FOR INSERT WITH CHECK (auth.uid() = profesor_id);

CREATE POLICY "Profesores can update their own exams" ON examenes
    FOR UPDATE USING (auth.uid() = profesor_id);

CREATE POLICY "Profesores can delete their own exams" ON examenes
    FOR DELETE USING (auth.uid() = profesor_id);

CREATE POLICY "Estudiantes can view published exams" ON examenes
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM usuarios
            WHERE id = auth.uid() AND rol = 'estudiante'
        )
    );

-- ===========================================
-- TABLA: preguntas
-- ===========================================
CREATE TABLE IF NOT EXISTS preguntas (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    examen_id UUID REFERENCES examenes(id) ON DELETE CASCADE,
    pregunta TEXT NOT NULL,
    opcion_a TEXT NOT NULL,
    opcion_b TEXT NOT NULL,
    opcion_c TEXT NOT NULL,
    opcion_d TEXT NOT NULL,
    respuesta_correcta CHAR(1) NOT NULL CHECK (respuesta_correcta IN ('a', 'b', 'c', 'd')),
    explicacion TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Habilitar RLS en preguntas
ALTER TABLE preguntas ENABLE ROW LEVEL SECURITY;

-- Políticas RLS para preguntas
CREATE POLICY "Profesores can manage questions of their exams" ON preguntas
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM examenes
            WHERE id = examen_id AND profesor_id = auth.uid()
        )
    );

CREATE POLICY "Estudiantes can view questions during exam" ON preguntas
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM usuarios
            WHERE id = auth.uid() AND rol = 'estudiante'
        )
    );

-- ===========================================
-- TABLA: resultados_examen
-- ===========================================
CREATE TABLE IF NOT EXISTS resultados_examen (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    examen_id UUID REFERENCES examenes(id) ON DELETE CASCADE,
    estudiante_id UUID REFERENCES usuarios(id) ON DELETE CASCADE,
    puntuacion INTEGER NOT NULL,
    total_preguntas INTEGER NOT NULL,
    tiempo_completado INTEGER, -- en segundos
    respuestas JSONB, -- almacenar las respuestas del estudiante
    fecha_completado TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Habilitar RLS en resultados_examen
ALTER TABLE resultados_examen ENABLE ROW LEVEL SECURITY;

-- Políticas RLS para resultados_examen
CREATE POLICY "Estudiantes can view their own results" ON resultados_examen
    FOR SELECT USING (auth.uid() = estudiante_id);

CREATE POLICY "Profesores can view results of their exams" ON resultados_examen
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM examenes
            WHERE id = examen_id AND profesor_id = auth.uid()
        )
    );

CREATE POLICY "Estudiantes can insert their own results" ON resultados_examen
    FOR INSERT WITH CHECK (auth.uid() = estudiante_id);

-- ===========================================
-- TABLA: evaluaciones_pdf
-- ===========================================
CREATE TABLE IF NOT EXISTS evaluaciones_pdf (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    profesor_id UUID REFERENCES usuarios(id) ON DELETE CASCADE,
    estudiante_id UUID REFERENCES usuarios(id) ON DELETE CASCADE,
    nombre_archivo VARCHAR(255) NOT NULL,
    contenido_analizado JSONB,
    resumen TEXT,
    calificacion_promedio DECIMAL(3,2),
    fecha_evaluacion TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Habilitar RLS en evaluaciones_pdf
ALTER TABLE evaluaciones_pdf ENABLE ROW LEVEL SECURITY;

-- Políticas RLS para evaluaciones_pdf
CREATE POLICY "Profesores can view evaluations they created" ON evaluaciones_pdf
    FOR SELECT USING (auth.uid() = profesor_id);

CREATE POLICY "Estudiantes can view their own evaluations" ON evaluaciones_pdf
    FOR SELECT USING (auth.uid() = estudiante_id);

CREATE POLICY "Profesores can create evaluations" ON evaluaciones_pdf
    FOR INSERT WITH CHECK (auth.uid() = profesor_id);

-- ===========================================
-- TABLA: sesiones_estudio
-- ===========================================
CREATE TABLE IF NOT EXISTS sesiones_estudio (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    estudiante_id UUID REFERENCES usuarios(id) ON DELETE CASCADE,
    materia VARCHAR(100) NOT NULL,
    tiempo_estudiado INTEGER NOT NULL, -- en minutos
    preguntas_practicadas INTEGER DEFAULT 0,
    fecha_sesion DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Habilitar RLS en sesiones_estudio
ALTER TABLE sesiones_estudio ENABLE ROW LEVEL SECURITY;

-- Políticas RLS para sesiones_estudio
CREATE POLICY "Estudiantes can manage their own study sessions" ON sesiones_estudio
    FOR ALL USING (auth.uid() = estudiante_id);

-- ===========================================
-- FUNCIONES ÚTILES
-- ===========================================

-- Función para actualizar updated_at automáticamente
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc'::text, NOW());
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers para updated_at
CREATE TRIGGER update_usuarios_updated_at BEFORE UPDATE ON usuarios
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_examenes_updated_at BEFORE UPDATE ON examenes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===========================================
-- ÍNDICES PARA MEJOR PERFORMANCE
-- ===========================================

CREATE INDEX IF NOT EXISTS idx_usuarios_cedula ON usuarios(cedula);
CREATE INDEX IF NOT EXISTS idx_usuarios_email ON usuarios(email);
CREATE INDEX IF NOT EXISTS idx_examenes_profesor ON examenes(profesor_id);
CREATE INDEX IF NOT EXISTS idx_examenes_materia ON examenes(materia);
CREATE INDEX IF NOT EXISTS idx_preguntas_examen ON preguntas(examen_id);
CREATE INDEX IF NOT EXISTS idx_resultados_estudiante ON resultados_examen(estudiante_id);
CREATE INDEX IF NOT EXISTS idx_resultados_examen ON resultados_examen(examen_id);
CREATE INDEX IF NOT EXISTS idx_evaluaciones_profesor ON evaluaciones_pdf(profesor_id);
CREATE INDEX IF NOT EXISTS idx_evaluaciones_estudiante ON evaluaciones_pdf(estudiante_id);
CREATE INDEX IF NOT EXISTS idx_sesiones_estudiante ON sesiones_estudio(estudiante_id);
