      PROGRAM lec_dmt_n1_1132_1134_1137

C-------------------------------------------------------------------------
C
C     DEMETER : DMT_N1_1132, DMT_N1_1134 and DMT_N1_1137 Datasets archived
c     in CDPP
C
c     Issue 01 - Rev 00
c     Date : 2007-04-26
C
c     This FORTRAN program reads DMT_N1_1132_... , DMT_N1_1134_... and
c     DMT_N1_1137_... .DAT files and prints the first record, the last
c     record and the number of records.
c
c     It can be compiled with GNU. It runs only on platforms supporting
c     Big Endian encoding.
c
c     The input file (file to be read) must be called "dmt_n1"
c     The output file is called "ed_dmt_n1"
C
C-------------------------------------------------------------------------
c

c--------------------------------------
c     Variables declarations
c--------------------------------------

c     Blocks 1, 2 et 3

      BYTE         vers1(4),vers2(2),vers3(2) 
      INTEGER*2    idatdmt(7),norb(2),qi
      INTEGER*4    datcds(2),day,msec
      REAL*4       orbp(4),geop(15),solp(3),attp(18)
      CHARACTER*8  tms

c     Block 4 : header

      BYTE         hk(32),nbsp
      INTEGER*2    nbf,idatsp(7)
      REAL*4       freq,frange(2),dt
      CHARACTER*3  name
      CHARACTER*9  coord
      CHARACTER*16 unit
      CHARACTER*21 type

c     Block 4 : data

      REAL*4       sp(2048)

c     Annexe

      INTEGER*2    nk,nd,nf


c------------------------------------------------------------
c     Opening the input and output files
c------------------------------------------------------------

      OPEN(1,FORM='UNFORMATTED',FILE='dmt_n1',ACCESS='DIRECT',
     &       RECL=8510)
      OPEN(3,FORM='FORMATTED',FILE='ed_dmt_n1')

C----------------------------------------------------------
C     Initializations
C----------------------------------------------------------

      nb_enr=0
      nlec=0

C----------------------------------------------------------
C     Reading a DMT_N1_1132 (or DMT_N1_1134 or DMT_N1_1137) data record
C----------------------------------------------------------

100   CONTINUE
      nlec=nlec+1
      READ(1,REC=nlec,IOSTAT=iostat) datcds,idatdmt,norb,tms,vers1,
     &       orbp,geop,solp,vers2,
     &       attp,qi,vers3,
     &       type,hk,coord,name,unit,nbsp,nbf,dt,freq,frange,idatsp,
     &       sp

c     Testing IO status (0 : OK ; -1 : end of file ; else : error)

      if (iostat.eq.0) then
        nb_enr=nb_enr+1
        if(nb_enr.ne.1) go to 100
      elseif (iostat.eq.-1) then
         go to 199
      else
        write(3,9998) nrec
        stop
      endif

199   continue

C------------------------------------------------------------------
C     Printing the first and the last data record
C------------------------------------------------------------------

c     Decoding CCSDS Time
c       datcds(1) contains P-field (1 byte) + days since 01-01-1950 (3 bytes)
c       datcds(2) contains milliseconds in day
      day=mod(datcds(1),16777216)
      msec=datcds(2)

c     Printing the record type (first or last)
      if (iostat.eq.0) then
         write(3,4000)
      elseif (iostat.eq.-1) then
         write(3,4100)
      endif

c     Printing the data
      write(3,1000) day,msec,idatdmt,norb,tms,vers1,
     &       orbp,geop,solp,vers2,
     &       attp,qi,vers3
      write(3,2000) type,hk,coord,name,unit,nbsp,nbf,dt,freq,frange,
     &       idatsp
      nk=nbsp
      do 200 k=1,nk
      nd=nbf*(k-1)+1
      nf=nbf*k
      write(3,3000) (sp(i),i=nd,nf)
200   continue

      if (iostat.eq.0) then
         go to 100
      elseif (iostat.eq.-1) then
         write(3,9000) nb_enr
         stop
      endif

C----------------------
C     Formats
C----------------------

1000  FORMAT(//,7x,i5,2x,i8,5x,i4.4,2('-',i2.2),'T',i2.2,
     1          2(':',i2.2),'.',i3.3,5x,i5.5,i1,3x,a8,3x,3i1,i2.2,
     2        /,4(/,5x,5e15.7),/,5x,2e15.7,13x,2i1,
     3        /,3(/,5x,5e15.7),/,5x,3e15.7,3x,i7,3x,2i1)
2000  FORMAT(/,7x,a21,/,7x,32z2.2,1x,a9,
     1       /,7x,a3,9x,a16,5x,2i5,
     2       /,5x,4e15.7,
     3       /,7x,i4,5(1x,i2.2),1x,i3.3)
3000  FORMAT(//,(5x,5e15.7))
4000  FORMAT(////,4X,76('*'),
     1         //,4X,'Contents of the first record',
     2         //,4X,76('*'))
4100  FORMAT(////,4X,76('*'),
     1         //,4X,'Contents of the last record',
     2         //,4X,76('*'))
9000  FORMAT(////,4X,'Number of data records = ',I6)
9998  FORMAT(/,4X,'Reading error on "dmt_n1" file',
     1       /,4X,'Record #',i6)

      END
