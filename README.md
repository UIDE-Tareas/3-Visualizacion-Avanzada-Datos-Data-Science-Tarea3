<div style="display: table; width: 100%;">
  <div style="display: table-cell; text-align: center; vertical-align: middle; width: 70%;">
    <h1>Visualización Avanzada de Datos en Data Science</h1>
  </div>
  <div style="display: table-cell; text-align: center; vertical-align: middle; width: 30%;">
    <img src="https://github.com/UIDE-Tareas/3-Visualizacion-Avanzada-Datos-Data-Science-Tarea3/blob/main/Assets/UideLogo.png?raw=true" alt="logo UIDE" style="width:50%;">
  </div>
</div>
<hr />

### 🟦 Componente Práctico 3  
🟡 Grupo: 4      
🟡 Semana: 3      
🟡 Docente: Edwin Jahir Rueda Rojas(edruedaro@uide.edu.ec)     

### 🟦 Realizado por:   
Estudiantes

💻 Diego Fernando Chimbo Yepez   

💻 Hugo Javier Erazo Granda

💻 José Espinoza Bone

<hr />

## Pasos para utilizar

### 1 Clonar este repositorio

En Windows, Linux o MacOS ejecutar el comando

```
git clone "https://github.com/UIDE-Tareas/3-Visualizacion-Avanzada-Datos-Data-Science-Tarea3.git"
```

### 2 Ejecutar el script App

Dependiendo de tu plataforma ejecutar el script App, este script permite crear el virtual environment para instalar las dependencias en este entorno y evitar que se instalen en la instalación de Python. Con esto evitamos actualizar o realizar cambios en la instalación general de python sin afectar otros proyectos o instalar dependencias innecesarias.

##### Windows

##### Desde Command Prompt (cmd.exe)
```cmd
App.bat
```

###### Desde PowerShell (PowerShell.exe, pwsh.exe)
```powershell
./App.ps1
```

#### Linux y MacOS

###### Desde la terminal ejecutar
```bash
source App
```
o

```bash
bash App
```


## Info adicional

#### Interfaz gráfica

Cada servidor de dashboard Flask/Dash se ejecuta con un puerto específico con el botón Ejecutar.

Para ver el respectivo dashboard, dar click en Navegar. Esto abre el navegador en una url http://host:port

Para detener el servidor Flask/Dash del dashboard, dar click en Detener

![Windows App](https://github.com/UIDE-Tareas/3-Visualizacion-Avanzada-Datos-Data-Science-Tarea3/blob/main/Screenshots/WindowsApp.png?raw=true)

![Linux App](https://github.com/UIDE-Tareas/3-Visualizacion-Avanzada-Datos-Data-Science-Tarea3/blob/main/Screenshots/LinuxApp.png?raw=true)


#### App.py 

- Descarga el archivo de datos csv.gz a la carpeta temporal y descomprime

- Descarga el css custom del dashboard 1 en la carpeta temporal

- Instala las dependencias

- Crear la interfaz gráfica con customtkinter

- Muestra la interfaz gráfica con las opciones.



#### DashApp1.py, DashApp2.py, DashApp3.py

Inicia la instancia del servidor Dash/Flash en su respectivo host y port, para ser accedido desde los clientes por medio del navegador.



#### App

Archivo para ser ejecutado en sistemas operativos Linux y MacOS con el comando `source App`



#### App.bat

Archivo para ser ejecutado en el command prompt de Windows  `cmd.exe`



#### App.ps1

Archivo para ser ejecutado en la shell de PowerShell  `Powershell.exe, pwsh.exe`




