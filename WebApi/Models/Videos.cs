﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.IO;
using System.Configuration;

namespace WebAPIFiles.Models
{
    public class videos_atributos
    {
        public string nombre { get; set; }
        public string Fecha { get; set; }
    }
    public class Videos
    {
        public string directorio_princ = ConfigurationManager.AppSettings["DirectorioPrincipal"];
        public List<videos_atributos> getAllVideos(string estacion, string fecha)
        {
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
                        lista_videos.Add(new videos_atributos() { nombre = nombre_video, Fecha = fecha_video });

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