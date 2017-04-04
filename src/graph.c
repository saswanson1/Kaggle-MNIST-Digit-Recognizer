/*-------------------------------------------------------------------------------------------------------------------*/
/* "Study hard what interests you the most in the most undisciplined, irreverent and original manner possible."      */
/*   -- Richard Feynman                                                                                              */
/*-------------------------------------------------------------------------------------------------------------------*/
/* REVISION HISTORY                                                                                                  */
/*                                                                                                                   */
/* 27 MAR 2017  SAS. Added file path argument to quickGraph().                                                       */
/* 25 MAR 2017  SAS. Initial creation, split out from psds.c.                                                        */
/*                                                                                                                   */
/*-------------------------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "graph.h"


/*********************************************************************************************************************/
/*********************************************************************************************************************/
void GRF_setColor( graphInfo_t *g, const char color[] )
{
  strncpy( (char *)g->color, color, sizeof( g->color ) );
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void GRF_setRefresh( graphInfo_t *g, const int refresh )
{
  g->refresh = refresh;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void GRF_setFilePath( graphInfo_t * g, const char path[] )
{
  strncpy( g->filePath, path, sizeof( g->filePath ) );
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void GRF_setXdata( graphInfo_t * graph, const double * x )
{
  graph->x = x;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void GRF_setYdata( graphInfo_t * graph, const double * y )
{
  graph->y = y;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
graphInfo_t * GRF_createGraph( void )
{
  graphInfo_t * g = malloc( sizeof( graphInfo_t ) );
  if( g )
  {
    g->state = SGE_DEFAULT;
    g->refresh = 0; /* default to no automatic refresh */
    g->filePath[0] = 0;
    g->title[0] = 0;
    g->x = NULL;
    g->y = NULL;
    g->n = 0;
  }
  return g;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
void GRF_releaseGraph( graphInfo_t * graph )
{
  if( graph )
    free( graph );
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
int GRF_generate( graphInfo_t *g )
{
  int i;
  FILE *f;

  if( (NULL == g->x) || (NULL == g->y) || (0 == g->n ) )
    return -1;

  /* Find min & max Y values in the data */
  g->minX = g->x[0];
  g->maxX = g->x[0];
  g->minY = g->y[0];
  g->maxY = g->y[0];
  for( i=1; i<g->n; i++ )
  {
    if( g->x[i] < g->minX )
      g->minX = g->x[i];
    else if( g->x[i] > g->maxX )
      g->maxX = g->x[i];
    if( g->y[i] < g->minY )
      g->minY = g->y[i];
    else if( g->y[i] > g->maxY )
      g->maxY = g->y[i];
  }

  g->spanX = g->maxX - g->minX;
  g->spanY = g->maxY - g->minY;

  /* If default graph size was specified, choose default size */
  if( g->graphHgt < 1.0 )
    g->graphHgt = 500;
  if( g->graphWid < 1.0 )
    g->graphWid = 900;

  g->xScale = ((double)g->graphWid)/(g->maxX - g->minX);
  g->yScale = ((double)g->graphHgt)/(g->maxY - g->minY);

  f = fopen( g->filePath, "w" );
  if( f )
  {
    fprintf( f, "<html>\n" );
    fprintf( f, "<head>\n" );
    fprintf( f, "<script>\n" );
    fprintf( f, "window.addEventListener( 'load', onloadHandler, false );\n" );
    fprintf( f, "function onloadHandler()\n" );
    fprintf( f, "{\n" );
    fprintf( f, "  var ctx = document.getElementById('canvas').getContext('2d');\n" );
    fprintf( f, "  ctx.translate( 60, 20 );\n" );

    /* Draw gridlines */
    fprintf( f, "  ctx.strokeStyle=\"#808080\";\n" );
    fprintf( f, "  ctx.lineWidth=0.5;\n" );
    for( i=0; i<=10; i++ )
    {
      /* vertical lines */
      fprintf( f, "  ctx.beginPath();\n" );
      fprintf( f, "  ctx.moveTo( %d, %d, 0 );\n",
                     (int)((double)i * (g->graphWid/10.0)), 0 );
      fprintf( f, "  ctx.lineTo( %d, %d, 0 );\n",
                     (int)((double)i * (g->graphWid/10.0)), (int)g->graphHgt );
      fprintf( f, "  ctx.stroke();\n" );

      /* horizontal lines */
      fprintf( f, "  ctx.beginPath();\n" );
      fprintf( f, "  ctx.moveTo( %d, %d, 0 );\n",
                     0, (int)g->graphHgt-(int)((double)i*(g->graphHgt/10.0)) );
      fprintf( f, "  ctx.lineTo( %d, %d, 0 );\n",
                     (int)g->graphWid, (int)g->graphHgt-(int)((double)i*(g->graphHgt/10.0)) );
      fprintf( f, "  ctx.stroke();\n" );

      /* vertical labels */
      fprintf( f, "  ctx.fillText( \"%.4lf\", %d, %d );\n",
                     g->minY+((double)i*(g->spanY/10.0)),
                     -40,  (int)g->graphHgt+4-(int)((double)i * (g->graphHgt/10.0)) );
      /* horizontal labels */
      fprintf( f, "  ctx.fillText( \"%.4lf\", %d, %d );\n",
                     g->minX + ((double)i * (g->spanX/10.0)),
                     (int)-13+(int)((double)i * (g->graphWid/10.0)),
                     (int)g->graphHgt+18 );
    }

    /* Graph the actual data */
    if( NULL == g->color )
      fprintf( f, "  ctx.strokeStyle=\"#000000\";\n" );
    else
      fprintf( f, "  ctx.strokeStyle=\"%s\";\n", g->color );

    fprintf( f, "  ctx.lineWidth=1.0;\n" );
    fprintf( f, "  ctx.beginPath();\n" );
    for( i=0; i<g->n; i++ )
    {
      if( 0 == i )
        fprintf( f, "  ctx.moveTo( %d, %d, 0 );\n",
                       (int)((g->x[i] - g->minX) * g->xScale),
                       (int)(g->graphHgt-((g->y[i] - g->minY) * g->yScale)) );
      else
        fprintf( f, "  ctx.lineTo( %d, %d, 0 );\n",
                       (int)((g->x[i] - g->minX) * g->xScale),
                       (int)(g->graphHgt - ((g->y[i] - g->minY) * g->yScale)) );
    }
    fprintf( f, "  ctx.stroke();\n" );

    /* Draw graph points on top of the line itself */
    for( i=0; i<g->n; i++ )
    {
      fprintf( f, "  ctx.beginPath();\n" );
      fprintf( f, "  ctx.arc( %d, %d, 3, 0, 2*Math.PI );\n",
                     (int)((g->x[i] - g->minX ) * g->xScale),
                     (int)(g->graphHgt-((g->y[i] - g->minY) * g->yScale)) );
      fprintf( f, "  ctx.stroke();\n" );
    }

    if( g->refresh )
      fprintf( f, "setTimeout(function(){ window.location.reload(1); }, %d);\n", g->refresh );

#if( 0 )
    fprintf( f, "addEventListener(\"click\", function()\n"
                "{\n"
                "  var canvas = document.getElementById('canvas');\n"
                "  var ctx = canvas.getContext('2d');\n"
                "  var rect = canvas.getBoundingClientRect();\n"
                "  var x =  event.clientX - rect.left - 60;\n"
                "  var y =  event.clientY - rect.top - 20;\n"
                "  ctx.fillText( x + \"  \" + (%d - y), x, y );\n"
                "} );\n", (int)graphHgt );
#endif

    fprintf( f, "}\n" );
    fprintf( f, "</script>\n" );
    fprintf( f, "</head>\n" );
    fprintf( f, "<body>\n" );
    fprintf( f, "<canvas id=\"canvas\" width=%d height=%d style=\"background-color:#F0F0F0\"></canvas>\n",
                (int)(g->graphWid + g->graphOrigX + 20), (int)(g->graphHgt + g->graphOrigY + 40 ) );
    fprintf( f, "</body>\n" );
    fprintf( f, "</html>\n" );
    fclose( f );
    f = NULL;
  }
  else
  {
    printf( "UNABLE TOP OPEN GRAPH FILE!\n" );
  }
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
int GRF_quickGraph
(
  const char filePath[], /* output file path; if NULL, default name "graph.html" will be used */
  const double x[],
  const double y[],
  int          n,        /* number of x/y pairs */
  const char   color[],
  double       graphWid, /* desired graph width, in pixels */
  double       graphHgt, /* desired graph height, in pixels */
  int          refresh   /* Auto reload interval in ms; pass zero for no auto reload */
)
{
  int status = 0;
  graphInfo_t *g = GRF_createGraph();
  if( NULL == g )
  {
    status = -1;
    goto qgCleanupAndExit;
  }

  if( filePath )
  {
    GRF_setFilePath( g, filePath );
  }
  else
  {
    GRF_setFilePath( g, "graph.html" );
  }
  GRF_setXdata( g, x );
  GRF_setYdata( g, y );
  g->n = n;
  GRF_setColor( g, color );
  g->graphWid = graphWid;
  g->graphHgt = graphHgt;
  GRF_setRefresh( g, refresh ); /* Set refresh (reload) interval */
  g->graphOrigX=60;
  g->graphOrigY=30;

  GRF_generate( g );

qgCleanupAndExit:
  if( NULL != g )
    GRF_releaseGraph( g );
  return status;
}





