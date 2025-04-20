---
title: 简单讲一下使用MS3D为opengl建模
author: sword865
type: post
date: 2011-03-16
url: /archives/124
posturl_add_url:
  - yes
categories:
  - 未分类
tags:
  - 3d
  - opengl
---
做毕设的时候写的东西，贴上来吧…………

由于在OPENGL只能通过程序语言绘制模型，远不能达到可见既可得的目的。因此，比起3DMAX、MAYA等可视化3D建模工具，OPENGL模型的建立就相当的困难，为了简化这一问题的处理，可以使用简单小巧的MS3D来完成可见即可得的绘制过程。

MS3D的文件有着非常简单良好的文件结构，可从该文件中完美读取在可视工具中绘制的3D图形模型包含的点、线、面等各项基本结构的参数与位置，并在OPENGL根据读取结果即可进行绘制重现该模型。

MS3D全名为MilkShape3D，是一款简单小巧的3D可视化图形建模工具，可以简单的使用各种点、线面等基本图形元素组合建立模型，并进行贴图，分组。进一步的，该工具还支持简单的骨骼动画制作，是一款非常好用的3D图形构建工具。

[<img class="alignnone  wp-image-125" src="http://blog.sword865.com/wp-content/uploads/2015/02/ms3d-300x164.jpg" alt="ms3d" width="428" height="234" />][1]

在建立了MS3D中完成模型建立后可保存为.ms3d的文件格式，通过对该文件格式进行分析，就可以了解文件结构，以在程序中通过读取该文件重现所见模型。

该文件依次包括6段信息，除第一段文件头外，其它每段的开始位置都记录了该段中元素的数目，可用于计算该段的具体大小。

  * 文件头:大小固定为14字节。前10个字节为固定的标志 MS3D000000<-其中后6个字节就是字符0（即值为48）后4个字节为该模型格式的版本号，这4个字节为一个有符号整数，目前该版本号的值为3或4，两种版本的格式细节不同。

  * 点数据：紧接着文件头的就是模型的顶点数据部分，顶点部分的头两个字节为一个无符号整数，表示有多少个顶点。之后便是一个接一个的顶点的具体数据，包括可见性，x,y,z的坐标和绑定骨骼的ID编号(未绑定骨骼则为-1)。

  * 多边形数据: 紧接着顶点数据的是多边形数据（三角形），多边形部分头两个字节是一个无符号整数，表示有多少个三角形。之后便是一个接一个的三角形数据。主要记录了每个三角形结构，包括顶点索引，顶点法线（用于光照计算），纹理坐标和组信息。

  * 组信息：即网格信息，出于灵活性的考虑，模型的一个个三角形被按照网格或是组来划分。网格部分头两个字节是一个无符号整数，表示有得多少个网格。之后便是一个接一个的网格数据，每个网格结构的大小可能不同（因为他们拥有的三角形数不同）。主要包括网格的名字（字符串），三角形数量、三角形索引和材质索引（无材质则为-1）。

  * 材质信息：贴图、颜色等材质部分。头两个字节是一个无符号整数，表示有多少个材质。之后便是一个接一个的材质信息。包括材质名、环境光、漫射光、高光、自发光、发光值、透明度、贴图文件名、透明贴图文件名。

  * 骨骼信息： 动画、动作等。该结构是MS3D中的动态结构，仅当建立动态动画时存在，包括一种名为关键帧的结构，记录时间与对应的坐标系变换。骨骼信息，一开始是两个字节的无符号整数，表示一共有多少个骨骼，之后便是一个个的骨骼，骨骼的大小不是固定的。主要包括了骨骼名字，父骨骼名字，初始旋转与初始平移、以及之后的各个旋转与平移关键帧。

在分析了解了MS3D的文件格式后，就可以通过编写程序读取MS3D文件并根据该文件建立模型了，对应于MS3D的不同分段，可以依次建立6种结构体分别对应每段内容：

    MS3DHeader     /\*包含ms3d文件的版本信息\*
    MS3DVertex     /\*顶点信息\*/
    MS3DMaterial   /\*材质(纹理贴图等)信息\*/
    MS3DTriangle   /\*绘制三角形信息\*/
    MS3DJoint      /\*节点(骨骼)信息\*/
    MS3DKeyframe   /\*关键窗口\*/
    //an example for vertex
    struct MS3DVertex
    {
      unsigned char m_ucFlags;   //编辑器用标志
      CVector3 m_vVert;        //x,y,z的坐标
      char m_cBone;        //Bone ID （-1 ,没有骨头）
      unsigned char m_mcUnused;      //保留，未使用
    };

(1)第一个成员表示了该顶点在编辑器中的状态（引擎中不是必须）其各个值的含义如下：

0：顶点可见，未选中状态

1：顶点可见，选中状态

2：顶点不可见，未选中状态

3：顶点不可见，选中状态

(2)第二个成员为顶点的坐标，CVector3为三个float型组成，总共12字节

(3)第三个成员为该顶点所绑定的骨骼的ID号，如果该值为-1 则代表没有绑定任何骨骼（静态）

(4)第四个成员不包含任何信息，直接略过。

将MS3D各段内容分别导入对应的结构体，将其读入内存。

多边形（三角形）结构读取示范：

    //内存空间分配
    // pPtr为文件读取偏移指针
    int nTriangles = *( word* )pPtr;
    m_numTriangles = nTriangles;
    m_pTriangles = new Triangle[nTriangles];
    pPtr += sizeof( word );
    //读取每个三角型
    for ( i = 0; i &lt; nTriangles; i++ )
    {
      MS3DTriangle *pTriangle = ( MS3DTriangle* )pPtr;
      int vertexIndices[3] = { pTriangle-&gt;m_vertexIndices[0], pTriangle-&gt;m_vertexIndices[1], pTriangle-&gt;m_vertexIndices[2] };
      float t[3] = { 1.0f-pTriangle-&gt;m_t[0], 1.0f-pTriangle-&gt;m_t[1], 1.0f-pTriangle-&gt;m_t[2] };
      //数据读取
      memcpy( m_pTriangles[i].m_vertexNormals, pTriangle-&gt;m_vertexNormals, sizeof( float )*3*3 );
      memcpy( m_pTriangles[i].m_s, pTriangle-&gt;m_s, sizeof( float )*3 );
      memcpy( m_pTriangles[i].m_t, t, sizeof( float )*3 );
      memcpy( m_pTriangles[i].m_vertexIndices, vertexIndices, sizeof( int )*3 );
      //文件读取指针前进
      pPtr += sizeof( MS3DTriangle );
    }

要注意得是，因为MS3D使用窗口坐标系而OpenGL使用笛卡儿坐标系，所以需要反转每个顶点Y方向的纹理坐标

除了读取模型信息外，还需要根据材质信息段中各种材质的贴图文件路径，读入对应的贴图文件，用以完成贴图。

通过对上一步中得到的各种结构体进行解析，可以得到以下以一些简单的基础结构：

Mesh   /\*网格\*/

Material  /\*材质\*/

Triangle  /\*三角形\*/

Vertex   /\*定点\*/

不同于以上结构体严格对应了MS3D各段存储结构，这些结构体仅仅包含了简单的图形信息，非常方便之后的绘制操作。

在读入MS3D文件后，可以使用OPENGL函数根据读入的数据与模型的信息，按网格分组，分别绘制每一组的数据。

    //按网格分组绘制
    for ( int i = 0; i &lt; m_numMeshes; i++ )
    {
      //材质，贴图等
      ………………………
      //开始绘制多边形（三角形）
      glBegin( GL_TRIANGLES );
      {
        for ( int j = 0; j &lt; m_pMeshes[i].m_numTriangles; j++ ){
        ……………………….
        //三角形所有顶点绘制
        for ( int k = 0; k &lt; 3; k++ )
        {
          //单个点的绘制
          ………………….
        }
      }
      glEnd();
    }

通过这种方法，就可以在程序中绘制一个个的具体模型：

<img class="alignnone  wp-image-127" src="http://blog.sword865.com/wp-content/uploads/2015/02/ballfight-300x168.jpg" alt="ballfight" width="352" height="197" />

&nbsp;

[1]<http://blog.sina.com.cn/s/blog_62d98a550100g5hh.html>

[2]<http://www.yakergong.net/nehe/course/tutorial_31.html>


