#ifndef _GRAPH_H_
#define _GRAPH_H_


typedef enum
{
  SGE_DEFAULT = 0,
  SGE_SIMPLE
} graphStateEnum_t;

typedef enum
{
  DXL_DEFAULT = 0,  /* defaults to double */
  DXL_DHMS,         /* DAYS:HOURS:MINUTES:SECONDS, from doubles in unites of seconds */
  DXL_STR           /* strings specified manually */
} graphXlabelEnum_t;

typedef struct _graphRect_t
{
  double left, top, right, bottom;
} graphRect_t;

typedef struct _graphInfo_t
{
  graphStateEnum_t  state;
  graphXlabelEnum_t xLabelType;
  int               refresh;  /* Graph refresh interval, in seconds; 0=no refresh */
  char              filePath[256];  /* file pathname */
  char              title[256];  /* chart title */
  const double    * x;  /* pointer to x data array */
  const double    * y;  /* pointer to y data array */
  int               n;  /* Number of data samples in x and y */
  double            minX, maxX;
  double            minY, maxY;
  double            xScale, yScale;
  double            graphOrigX, graphOrigY;
  double            spanX, spanY;
  const char        color[64];  /*data plot color */
  double            graphWid, graphHgt; /* desired graph width & height, in pixels */
} graphInfo_t;


/* PROTOTPYES */
int           GRF_quickGraph( const char fPath[], const double x[], const double y[], int n, const char color[],
                          double graphWid, double graphHgt, int refresh );
graphInfo_t * GRF_createGraph( void );
void          GRF_releaseGraph( graphInfo_t * graph );
int           GRF_generate( graphInfo_t *gr );
void          GRF_setXdata( graphInfo_t * graph, const double * x );
void          GRF_setYdata( graphInfo_t * graph, const double * y );
void          GRF_setFilePath( graphInfo_t * graph, const char path[] );
void          GRF_setColor( graphInfo_t *g, const char color[] );
void          GRF_setRefresh( graphInfo_t *g, const int refresh );
#endif












