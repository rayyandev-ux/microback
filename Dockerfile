# Usa una imagen base ligera de Node.js
FROM node:20-alpine

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de definición de dependencias
COPY package.json package-lock.json* ./

# Instala las dependencias de producción
RUN npm ci --only=production

# Copia el resto del código de la aplicación
COPY . .

# Expone el puerto que usa la aplicación (por defecto 3000)
ENV PORT=3000
EXPOSE 3000

# Comando para iniciar la aplicación
CMD ["node", "server.js"]
