#include "sz_compression_utils.hpp"

void
encode_regression_coefficients_2d(const int * reg_params_type, const float * reg_unpredictable_data, size_t reg_count, size_t reg_unpredictable_count, unsigned char *& compressed_pos){
	write_variable_to_dst(compressed_pos, reg_unpredictable_count);
	write_array_to_dst(compressed_pos, reg_unpredictable_data, reg_unpredictable_count);
	Huffman_encode_tree_and_data(2*RegCoeffCapacity, reg_params_type, RegCoeffNum2d*reg_count, compressed_pos);
}

void
encode_regression_coefficients(const int * reg_params_type, const float * reg_unpredictable_data, size_t reg_count, size_t reg_unpredictable_count, unsigned char *& compressed_pos){
	write_variable_to_dst(compressed_pos, reg_unpredictable_count);
	write_array_to_dst(compressed_pos, reg_unpredictable_data, reg_unpredictable_count);
	Huffman_encode_tree_and_data(2*RegCoeffCapacity, reg_params_type, RegCoeffNum3d*reg_count, compressed_pos);
}

// copied from conf.c
unsigned int 
round_up_power_of_2(unsigned int base){
  base -= 1;
  base = base | (base >> 1);
  base = base | (base >> 2);
  base = base | (base >> 4);
  base = base | (base >> 8);
  base = base | (base >> 16);
  return base + 1;
} 

// modified from TypeManager.c
// change return value and increment byteArray
void 
convertIntArray2ByteArray_fast_1b_to_result_sz(const unsigned char* intArray, size_t intArrayLength, unsigned char *& compressed_pos){
	size_t byteLength = 0;
	size_t i, j; 
	if(intArrayLength%8==0)
		byteLength = intArrayLength/8;
	else
		byteLength = intArrayLength/8+1;
		
	size_t n = 0;
	int tmp, type;
	for(i = 0;i<byteLength;i++){
		tmp = 0;
		for(j = 0;j<8&&n<intArrayLength;j++){
			type = intArray[n];
			if(type == 1)
				tmp = (tmp | (1 << (7-j)));
			n++;
		}
    	*(compressed_pos++) = (unsigned char)tmp;
	}
}

void convertByteArray2IntArray_fast_1b_sz(size_t intArrayLength, const unsigned char *&compressed_pos, size_t byteArrayLength, unsigned char *intArray)
    { // num_elements is intArrayLength,compresed_pos is reading bytes, byteArrayLength is nume_elements/8.ceil, intArray is output(bit map)
	  
        if (intArrayLength > byteArrayLength * 8)
        {
            printf("Error: intArrayLength > byteArrayLength*8\n");
            printf("intArrayLength=%zu, byteArrayLength = %zu", intArrayLength, byteArrayLength);
            exit(0);
        }
        size_t n = 0, i;
        int tmp;
        for (i = 0; i < byteArrayLength - 1; i++)
        {
            tmp = *(compressed_pos++);
            intArray[n++] = (tmp & 0x80) >> 7;
            intArray[n++] = (tmp & 0x40) >> 6;
            intArray[n++] = (tmp & 0x20) >> 5;
            intArray[n++] = (tmp & 0x10) >> 4;
            intArray[n++] = (tmp & 0x08) >> 3;
            intArray[n++] = (tmp & 0x04) >> 2;
            intArray[n++] = (tmp & 0x02) >> 1;
            intArray[n++] = (tmp & 0x01) >> 0;
        }
        tmp = *(compressed_pos++);
        for (int i = 0; n < intArrayLength; n++, i++)
        {
            intArray[n] = (tmp & (1 << (7 - i))) >> (7 - i);
        }
    }



HuffmanTree *
build_Huffman_tree(size_t state_num, const int * type, size_t num_elements){
	HuffmanTree * huffman = createHuffmanTree(state_num);
	init(huffman, type, num_elements);
	return huffman;
}

void
Huffman_encode_tree_and_data(size_t state_num, const int * type, size_t num_elements, unsigned char*& compressed_pos){
	HuffmanTree * huffman = build_Huffman_tree(state_num, type, num_elements);
	size_t node_count = 0;
	size_t i = 0;
	for (i = 0; i < state_num; i++)
		if (huffman->code[i]) node_count++; 
	node_count = node_count*2-1;
	unsigned char *tree_structure = NULL;
	unsigned int tree_size = convert_HuffTree_to_bytes_anyStates(huffman, node_count, &tree_structure);
	write_variable_to_dst(compressed_pos, node_count);
	write_variable_to_dst(compressed_pos, tree_size);
	write_array_to_dst(compressed_pos, tree_structure, tree_size);
	unsigned char * type_array_size_pos = compressed_pos;
	compressed_pos += sizeof(size_t);
	size_t type_array_size = 0; 
	encode(huffman, type, num_elements, compressed_pos, &type_array_size);
	write_variable_to_dst(type_array_size_pos, type_array_size);
	compressed_pos += type_array_size;
	free(tree_structure);
	SZ_ReleaseHuffman(huffman);
}



