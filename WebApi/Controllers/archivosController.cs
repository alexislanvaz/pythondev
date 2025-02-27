﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using System.IO;
using System.Configuration;
using System.ComponentModel;
namespace WebAPIFiles.Controllers
{
    public class archivosController : ApiController
    {
        //public HttpResponseMessage Get(string nombre)
        //{
        //    HttpResponseMessage resultado = null;
        //    string directorio_princ = ConfigurationManager.AppSettings["DirectorioDocs"];


        //    int tipo_respuesta = 2;
        //    string msg_respuesta = "";
        //    byte[] contenido_array = null;
        //    try
        //    {
        //        if (Directory.Exists(directorio_princ))
        //        {

        //            string nombreCompletoArchivo = Path.Combine(directorio_princ, nombre);
        //            if (File.Exists(nombreCompletoArchivo))
        //            {
        //                long tamanio_video = new FileInfo(nombreCompletoArchivo).Length;
        //                contenido_array = new byte[tamanio_video];
        //                using (FileStream fs = new FileStream(nombreCompletoArchivo, FileMode.Open, FileAccess.Read, FileShare.Read))
        //                {
        //                    fs.Read(contenido_array, 0, contenido_array.Length);
        //                    tipo_respuesta = 1;
        //                }
        //            }
        //            else
        //            {
        //                msg_respuesta = "No existe el archivo seleccionado.";
        //            }
        //        }
        //        else
        //        {
        //            msg_respuesta = "No existe el directorio de descarga: ";
        //        }
        //    }
        //    catch(Exception ex)
        //    {
        //        tipo_respuesta = 3;
        //        msg_respuesta = ex.Message;
        //    }
        //    if (tipo_respuesta == 1)
        //        resultado = Request.CreateResponse(HttpStatusCode.OK, contenido_array);
        //    else
        //    {
        //        if (tipo_respuesta == 2)
        //            resultado = new HttpResponseMessage(HttpStatusCode.BadRequest);
        //        else
        //            resultado = new HttpResponseMessage(HttpStatusCode.InternalServerError);
        //        resultado.Content = new StringContent(msg_respuesta);
        //    }

        //    return resultado;

        //}


        public class videos_atributos
        {
            public string nombre { get; set; }
            public string fecha { get; set; }

        }
        public string directorio_princ = ConfigurationManager.AppSettings["DirectorioPrincipal"];
        public List<videos_atributos> Get(string ruta)
        {
            string estacion = ruta.Split('-')[0];
            string fecha = ruta.Split('-')[1];
            List<videos_atributos> lista_videos = new List<videos_atributos>();
            try
            {
                DirectoryInfo info_carpeta = new DirectoryInfo(directorio_princ + estacion + @"\" + fecha);
                FileInfo[] info_videos = info_carpeta.GetFiles("*.pdf");
                if (info_videos.Length > 0)
                {

                    foreach (FileInfo video in info_videos)
                    {
                        var nombre_video = video.Name;
                        var fecha_video = video.CreationTime.ToString("g");
                        lista_videos.Add(new videos_atributos() { nombre = nombre_video, fecha = fecha_video });

                    }
                }
            }
            catch
            {

            }
            return lista_videos.ToList();
        }
    }
}
