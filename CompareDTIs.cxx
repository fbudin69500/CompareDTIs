#include "CompareDTIsCLP.h"
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkDiffusionTensor3D.h>
#include <itkImageRegionIterator.h>
#include <limits>
#include <cmath>
#include "vnl/vnl_vector.h"
#include <sstream>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1


float Compute( float valueref , float value2 , std::string type )
{
  float temp ;
  if( !type.compare( "div" ) )
  {
    temp = value2 / valueref ;
  }
  else if( !type.compare( "sub" ) )
  {
    temp = valueref - value2 ;
  }
  else if( !type.compare( "subref" ) )
  {
    temp = ( valueref - value2 ) / valueref ;
  }
  else// subcomp
  {
    temp = ( valueref - value2 ) / ( valueref + value2 ) ;
  }
  return temp ;
}


float GetValue( itk::DiffusionTensor3D< float > tensor , std::string compare )
{
  float value = 0.0 ;
  if( !compare.compare( "FA" ) )
  {
    value = tensor.GetFractionalAnisotropy() ;
  }
  else if( !compare.compare( "MD" ) )
  {
    value = tensor.GetTrace() / 3 ;
  }
  return value ;
}


float CompareValues( itk::Image< itk::DiffusionTensor3D< float > , 3 >::Pointer ref ,
                  itk::Image< itk::DiffusionTensor3D< float > , 3 >::Pointer image2 ,
                  itk::Image< unsigned char , 3 >::Pointer mask ,
                  itk::Image< float , 3 >::Pointer &output ,
                  std::string compare ,
                  std::string type
                )
{
  typedef itk::Image< unsigned char , 3 > MaskType ;
  typedef itk::Image< float , 3 > ImageType ;
  typedef itk::DiffusionTensor3D< float > TensorType ;
  typedef itk::Image< TensorType , 3 > DTIType ;
  itk::ImageRegionIterator< DTIType > itref( ref , ref->GetLargestPossibleRegion() ) ;
  itk::ImageRegionIterator< DTIType > it2( image2 , image2->GetLargestPossibleRegion() ) ;
  itk::ImageRegionIterator< ImageType > *out = NULL ;
  itk::ImageRegionIterator< MaskType > *itmask = NULL ;
  if( mask.GetPointer() )
  {
    itmask = new itk::ImageRegionIterator< MaskType >( mask ,
                              mask->GetLargestPossibleRegion() ) ;
    itmask->GoToBegin() ;
  }
  if( output.GetPointer() )
  {
    out = new itk::ImageRegionIterator< ImageType >( output ,
                              output->GetLargestPossibleRegion() ) ;
    out->GoToBegin() ;
  }
  float sum  = 0.0 ;
  long count = 0 ;
  for( itref.GoToBegin() , it2.GoToBegin() ; !itref.IsAtEnd() ; ++itref , ++it2 )
  {
    if( !itmask || ( mask.GetPointer() && itmask->Get() ) )
    {
      float temp = 0.0 ;
      float valueref = GetValue( itref.Get() , compare ) ;
      float value2 = GetValue( it2.Get() , compare ) ;
      temp = Compute( valueref , value2 , type ) ;
      if( out )
      {
        out->Set( temp ) ;
      }
      if( !std::isnan( temp )
          && !( std::numeric_limits< float >::has_infinity
                && ( temp < 0 ? -temp : temp ) == std::numeric_limits< float >::infinity()
              )
        )
      {
        count++ ;//count the numbers of pixels in the mask or in the entire image if no mask
        sum += (temp < 0 ? -temp : temp ) ;
      }
    }
    if( itmask )
    {
      ++(*itmask) ;
    }
    if( out )
    {
       ++(*out) ;
    }
  }
  return sum / (float)count ; 
}

int StringToInt( std::string ev )
{
  int val ;
  std::istringstream stream( ev ) ;
  stream >> val ;
  return val ;
}

float CompareAngle( itk::Image< itk::DiffusionTensor3D< float > , 3 >::Pointer ref ,
                     itk::Image< itk::DiffusionTensor3D< float > , 3 >::Pointer image2 ,
                     itk::Image< unsigned char , 3 >::Pointer mask ,
                     itk::Image< float , 3 >::Pointer &output ,
                     std::string ev
                   )
{
  typedef itk::Image< unsigned char , 3 > MaskType ;
  typedef itk::Image< float , 3 > ImageType ;
  typedef itk::DiffusionTensor3D< float > TensorType ;
  typedef itk::Image< TensorType , 3 > DTIType ;
  itk::ImageRegionIterator< DTIType > itref( ref , ref->GetLargestPossibleRegion() ) ;
  itk::ImageRegionIterator< DTIType > it2( image2 , image2->GetLargestPossibleRegion() ) ;
  itk::ImageRegionIterator< ImageType > *out = NULL ;
  itk::ImageRegionIterator< MaskType > *itmask = NULL ;
  int vec = StringToInt( ev ) ;
  if( mask.GetPointer() )
  {
    itmask = new itk::ImageRegionIterator< MaskType >( mask ,
                              mask->GetLargestPossibleRegion() ) ;
    itmask->GoToBegin() ;
  }
  if( output.GetPointer() )
  {
    out = new itk::ImageRegionIterator< ImageType >( output ,
                              output->GetLargestPossibleRegion() ) ;
    out->GoToBegin() ;
  }
  float sum = 0.0 ;
  long count = 0 ;
  for( itref.GoToBegin() , it2.GoToBegin() ; !itref.IsAtEnd() ; ++itref , ++it2 )
  {
    if( !itmask || ( mask.GetPointer() && itmask->Get() ) )
    {
      TensorType::EigenValuesArrayType eigenValues ;
      TensorType::EigenVectorsMatrixType eigenVectors1 ;
      itref.Get().ComputeEigenAnalysis( eigenValues , eigenVectors1 ) ;
      TensorType::EigenVectorsMatrixType eigenVectors2 ;
      it2.Get().ComputeEigenAnalysis( eigenValues , eigenVectors2 ) ;
      typedef itk::Vector< double , 3 > VectorType ;
      VectorType vector1 ;
      VectorType vector2 ;
      for( int i = 0 ; i < 3 ; i++ )
      {
        vector1[ i ] = eigenVectors1[ i ][ vec ] ;
        vector2[ i ] = eigenVectors2[ i ][ vec ] ;
      }
      float angle_val = angle( vector1.GetVnlVector() , vector2.GetVnlVector() ) * 180 / M_PI;
      if( out )
      {
         out->Set( angle_val ) ;
      }
      count++ ;//count the numbers of pixels in the mask or in the entire image if no mask
      sum += (angle_val < 0 ? -angle_val : angle_val ) ;
    }
    if( itmask )
    {
      ++(*itmask) ;
    }
    if( out )
    {
       ++(*out) ;
    }
  }
  return sum / (float)count ; 
}


float Frobenius( itk::DiffusionTensor3D< float > tensor )
{
  float temp = 0.0 ;
  for( int i = 0 ; i < 3 ; i++ )
  {
    for( int j = 0 ; j < 3 ; j++ )
    {
      float local = tensor( i , j ) ;
      temp += local * local ;
    }
  }
  return sqrt( temp ) ;
}

float ComputeNorm( itk::DiffusionTensor3D< float > tensor1 ,
                    itk::DiffusionTensor3D< float > tensor2 ,
                    std::string type
                  )
{
  float temp = 0.0;
  if( !type.compare( "div" ) )
  {
    for( int i = 0 ; i < 3 ; i++ )
    {
      for( int j = 0 ; j < 3 ; j++ )
      {
        tensor2( i , j ) = tensor2( i , j ) / tensor1( i , j ) ;
      }
    }
    return Frobenius( tensor2 ) ;
  }
  else if( !type.compare( "sub" ) )
  {
    return Frobenius( tensor1 - tensor2 ) ;
  }
  else if( !type.compare( "subref" ) )
  {
    for( int i = 0 ; i < 3 ; i++ )
    {
      for( int j = 0 ; j < 3 ; j++ )
      {
        tensor2( i , j ) = ( tensor1( i , j ) - tensor2( i , j ) ) / tensor1( i , j ) ;
      }
    }    
    return Frobenius( tensor2 ) ;
  }
  else// subcomp
  {
    for( int i = 0 ; i < 3 ; i++ )
    {
      for( int j = 0 ; j < 3 ; j++ )
      {
        tensor2( i , j ) = ( tensor1( i , j ) - tensor2( i , j ) )
                         / ( tensor1( i , j ) + tensor2( i , j ) ) ;
      }
    }
    return Frobenius( tensor2 ) ;
  }
  return temp ;
}

float CompareFrobenius( itk::Image< itk::DiffusionTensor3D< float > , 3 >::Pointer ref ,
                         itk::Image< itk::DiffusionTensor3D< float > , 3 >::Pointer image2 ,
                         itk::Image< unsigned char , 3 >::Pointer mask ,
                         itk::Image< float , 3 >::Pointer &output ,
                         std::string type
                       )
{
  typedef itk::Image< unsigned char , 3 > MaskType ;
  typedef itk::DiffusionTensor3D< float > TensorType ;
  typedef itk::Image< TensorType , 3 > DTIType ;
  typedef itk::Image< float , 3 > ImageType ;
  itk::ImageRegionIterator< DTIType > itref( ref , ref->GetLargestPossibleRegion() ) ;
  itk::ImageRegionIterator< DTIType > it2( image2 , image2->GetLargestPossibleRegion() ) ;
  itk::ImageRegionIterator< ImageType > *out = NULL ;
  itk::ImageRegionIterator< MaskType > *itmask = NULL ;
  if( mask.GetPointer() )
  {
    itmask = new itk::ImageRegionIterator< MaskType >( mask ,
                              mask->GetLargestPossibleRegion() ) ;
    itmask->GoToBegin() ;
  }
  if( output.GetPointer() )
  {
    out = new itk::ImageRegionIterator< ImageType >( output ,
                              output->GetLargestPossibleRegion() ) ;
    out->GoToBegin() ;
  }
  float sum = 0.0 ;
  long count = 0 ;
  for( itref.GoToBegin() , it2.GoToBegin() ; !itref.IsAtEnd() ; ++itref , ++it2 )
  {
    if( !itmask || ( mask.GetPointer() && itmask->Get() ) )
    {
      float temp = ComputeNorm( itref.Get() , it2.Get() , type ) ;
      if( out )
      {
        out->Set( temp ) ;
      }
      if( !std::isnan( temp ) 
          && !( std::numeric_limits< float >::has_infinity 
                && ( temp < 0 ? -temp : temp ) == std::numeric_limits< float >::infinity()
              )
        )
      {
        count++ ;//count the numbers of pixels in the mask or in the entire image if no mask
        sum += (temp < 0 ? -temp : temp ) ;
      }
    }
    if( itmask )
    {
      ++(*itmask) ;
    }
    if( out )
    {
       ++(*out) ;
    }
  }
  return sum / (float)count ; 
}


int main( int argc , char** argv )
{
  PARSE_ARGS ;
  typedef itk::DiffusionTensor3D< float > TensorType ;
  typedef itk::Image< TensorType , 3 > DTIType ;
  typedef itk::ImageFileReader< DTIType > DTIReaderType ;
  //Read first DTI
  DTIReaderType::Pointer reader1 = DTIReaderType::New() ;
  reader1->SetFileName( inputVolume1 ) ;
  reader1->Update();
  DTIType::SizeType size1 ;
  size1 = reader1->GetOutput()->GetLargestPossibleRegion().GetSize() ;
  //read second DTI
  DTIReaderType::Pointer reader2 = DTIReaderType::New() ;
  reader2->SetFileName( inputVolume2 ) ;
  reader2->Update();
  DTIType::SizeType size2 ;
  size2 = reader2->GetOutput()->GetLargestPossibleRegion().GetSize() ;
  //Check that images have the same size
  for( int i = 0 ; i < 3 ; i++ )
  {
    if( size1[ i ] != size2[ i ] )
    {
      std::cerr << "DTIs should have the same size" << std::endl ;
      return EXIT_FAILURE ;
    }
  }
  //Create output image if necessary
  typedef itk::Image< float , 3 > ImageType ;
  ImageType::Pointer output = ImageType::New() ;
  if( outputVolume.compare("") )
  {
    output->CopyInformation( reader1->GetOutput() ) ;
    output->SetRegions( size1 ) ;
    output->Allocate() ;
    if( mask.compare( "" ) )
    {
       output->FillBuffer( 0.0 ) ;
    }
  }
  //Read mask if any
  typedef itk::Image< unsigned char , 3 > MaskType ;
  typedef itk::ImageFileReader< MaskType > MaskReaderType ;
  MaskReaderType::Pointer maskReader = MaskReaderType::New() ;
  if( mask.compare( "" ) )
  {
    maskReader->SetFileName( mask ) ;
    maskReader->Update() ;
  }
  float average = 0.0 ;
  if( !compare.compare("FA") || !compare.compare( "MD") )
  {
    average = CompareValues( reader1->GetOutput() ,
                             reader2->GetOutput() ,
                             maskReader->GetOutput() ,
                             output ,
                             compare ,
                             diff
                           ) ;
  }
  else if( !compare.compare("Frobenius") )
  {
    average = CompareFrobenius( reader1->GetOutput() ,
                                reader2->GetOutput() ,
                                maskReader->GetOutput() ,
                                output ,
                                diff
                              ) ;    
  }
  else
  {
    average = CompareAngle( reader1->GetOutput() ,
                            reader2->GetOutput() ,
                            maskReader->GetOutput() ,
                            output ,
                            ev
                          ) ;    
  }  
  std::cout << average << std::endl ;
  if( outputVolume.compare( "" ) )
  {
    typedef itk::ImageFileWriter< ImageType > WriterType ;
    WriterType::Pointer writer = WriterType::New() ;
    writer->SetFileName( outputVolume ) ;
    writer->SetInput( output ) ;
    writer->Update() ;
  }
  return EXIT_SUCCESS ;
}

