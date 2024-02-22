import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="간단한 계산기 프로그램",
        description='더하기 빼기 기능만 존재',
        epilog='chaeyun'
    )
    #parser.parse_args()
    parser.add_argument("-i", "--input", type=int, nargs=2, help="2개의 값을 입력하세요", required=True)
    parser.add_argument("-c", "--calc", choices=["+", "-"], help="+ 또는 - 입력", required=True)
    args = parser.parse_args()

    if args.calc == "+":
        print(args.input[0]+args.input[1])
    elif args.calc == "-":
        print(args.input[0]-args.input[1])